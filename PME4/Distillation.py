"""
PME4 Knowledge Distillation Training Script
Teacher : TeacherModel (Audio → EEG 指导)
Student : ST_GCLSTM   (仅 EEG 输入)
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from PME4.model.Teacher  import TeacherModel
from PME4.model.Student  import ST_GCLSTM
from PME4.dataset.dataset import CrossSubjectPME4Dataset


# =============================================================================
# 1. 配置
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root  = '/data2/zhiwen/dataset/PME4'
    normalize  = True

    # ── 模型（Teacher & Student 共享 EEG 结构参数）────────────────────────────
    num_nodes       = 8     # PME4 电极数
    eeg_in_features = 5     # DE 5 频带
    gcn_hidden      = 64
    gcn_out         = 64
    lstm_hidden     = 64
    lstm_layers     = 1
    fc_hidden       = 64
    num_classes     = 2
    dropout         = 0.5

    # Teacher 专用 Audio 参数
    audio_input_dim  = 25   # OpenSMILE eGeMAPS F=25
    audio_hidden_dim = 128
    audio_num_layers = 2
    audio_dropout    = 0.5
    dk               = 32

    # ── 蒸馏权重 ──────────────────────────────────────────────────────────────
    base_w_ce        = 1.0
    base_w_graph     = 2.0
    base_w_temporal  = 0.5
    base_w_logits    = 0.0
    temperature      = 1.0

    # 前期增强权重（线性过渡到 base）
    initial_w_graph    = 0.0
    initial_w_temporal = 0.0
    weight_decay_epoch = 0

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 5e-4
    weight_decay = 1e-3
    patience     = 20

    # ── 输出 ──────────────────────────────────────────────────────────────────
    save_dir = './checkpoints'
    exp_name = 'st_gclstm_distill_pme4'

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 2024


# =============================================================================
# 2. 蒸馏损失（与 EAV 版完全一致，直接复用）
# =============================================================================

class DistillationLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.base_w_ce         = cfg.base_w_ce
        self.base_w_graph      = cfg.base_w_graph
        self.base_w_temporal   = cfg.base_w_temporal
        self.base_w_logits     = cfg.base_w_logits
        self.initial_w_graph   = cfg.initial_w_graph
        self.initial_w_temporal= cfg.initial_w_temporal
        self.weight_decay_epoch= cfg.weight_decay_epoch
        self.T  = cfg.temperature
        self.ce = nn.CrossEntropyLoss()

    def get_weights(self, epoch: int) -> dict:
        if epoch < self.weight_decay_epoch:
            f = epoch / self.weight_decay_epoch
            w_graph    = self.initial_w_graph    * (1 - f) + self.base_w_graph    * f
            w_temporal = self.initial_w_temporal * (1 - f) + self.base_w_temporal * f
        else:
            w_graph    = self.base_w_graph
            w_temporal = self.base_w_temporal
        return {
            'w_ce':      self.base_w_ce,
            'w_graph':   w_graph,
            'w_temporal':w_temporal,
            'w_logits':  self.base_w_logits,
        }

    def forward(self, s_out: dict, t_out: dict,
                labels: torch.Tensor, epoch: int = 0) -> dict:
        w = self.get_weights(epoch)

        # Hard label CE
        l_ce = self.ce(s_out['logits'], labels)

        # Graph KL：S_attn 每行是 N 维概率分布
        B, T_seq, N, _ = s_out['S_attn'].shape
        s_S = s_out['S_attn'].reshape(-1, N)
        t_S = t_out['S_attn'].reshape(-1, N).detach()
        l_graph = F.kl_div(
            s_S.clamp(min=1e-8).log(),
            t_S.clamp(min=1e-8),
            reduction='sum',
        ) / (B * T_seq * N)

        # Temporal KL
        l_temporal = F.kl_div(
            s_out['attn_t'].clamp(min=1e-8).log(),
            t_out['attn_t'].detach().clamp(min=1e-8),
            reduction='batchmean',
        )

        # Logits KL
        t_soft     = F.softmax(t_out['eeg_logits'] / self.T, dim=-1).detach()
        s_log_soft = F.log_softmax(s_out['logits']  / self.T, dim=-1)
        l_logits   = F.kl_div(s_log_soft, t_soft,
                               reduction='batchmean') * (self.T ** 2)

        total = (w['w_ce']       * l_ce
               + w['w_graph']    * l_graph
               + w['w_temporal'] * l_temporal
               + w['w_logits']   * l_logits)

        return {
            'loss':      total,
            'l_ce':      l_ce,
            'l_graph':   l_graph,
            'l_temporal':l_temporal,
            'l_logits':  l_logits,
            'weights':   w,
        }


# =============================================================================
# 3. 对齐数据集（EEG + PCC + Audio，去掉 Vision）
# =============================================================================

class PME4DistillDataset(Dataset):
    """
    索引同步 EEG / PCC / Audio，保证 Teacher 和 Student 看到同一个样本。
    EEG 有缺失文件（9个），以 min(N_eeg, N_audio) 对齐。
    """
    def __init__(self,
                 eeg:   np.ndarray,
                 pcc:   np.ndarray,
                 audio: np.ndarray,
                 y:     np.ndarray):
        n = min(len(eeg), len(audio), len(y))
        if n < len(eeg):
            print(f"  ⚠️  对齐截断: eeg={len(eeg)}, audio={len(audio)} → {n}")
        self.eeg   = torch.tensor(eeg[:n],   dtype=torch.float32)
        self.pcc   = torch.tensor(pcc[:n],   dtype=torch.float32)
        self.audio = torch.tensor(audio[:n], dtype=torch.float32)
        self.y     = torch.tensor(y[:n],     dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.eeg[idx], self.pcc[idx], self.audio[idx], self.y[idx]


# =============================================================================
# 4. 数据加载
# =============================================================================

def build_loaders(cfg: Config):
    manager = CrossSubjectPME4Dataset(root=cfg.data_root, fs=1000)

    # EEG + PCC
    print("\n" + "="*60)
    print("Loading EEG + PCC ...")
    (tr_eeg, tr_ey, tr_pcc), (va_eeg, va_ey, va_pcc), (te_eeg, te_ey, te_pcc) = \
        manager.get_all_splits(
            segment_1s=True, extract_de=True,
            normalize=True, compute_pcc=True)

    # Audio（OpenSMILE）
    print("\n" + "="*60)
    print("Loading Audio (OpenSMILE) ...")
    (tr_aud, tr_ay), (va_aud, va_ay), (te_aud, te_ay) = \
        manager.get_all_splits(
            modality="audio_opensmile", normalize=True)

    def make_loader(eeg, pcc, aud, y, shuffle):
        ds = PME4DistillDataset(eeg, pcc, aud, y)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    return (make_loader(tr_eeg, tr_pcc, tr_aud, tr_ey, shuffle=True),
            make_loader(va_eeg, va_pcc, va_aud, va_ey, shuffle=False),
            make_loader(te_eeg, te_pcc, te_aud, te_ey, shuffle=False))


# =============================================================================
# 5. 工具
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


# =============================================================================
# 6. 单 epoch 训练 / 评估
# =============================================================================

def run_epoch(cfg, teacher, student, loader, criterion,
              optimizer=None, epoch: int = 0) -> dict:
    is_train = optimizer is not None
    student.train() if is_train else student.eval()
    teacher.eval()

    meters = {k: AverageMeter()
              for k in ('loss', 'l_ce', 'l_graph', 'l_temporal', 'l_logits')}
    all_preds, all_labels = [], []

    for eeg, pcc, audio, y in loader:
        eeg, pcc, audio, y = (
            eeg.to(cfg.device), pcc.to(cfg.device),
            audio.to(cfg.device), y.to(cfg.device))

        # Teacher forward（固定权重，无梯度）
        with torch.no_grad():
            t_out = teacher(eeg, pcc, audio)   # ← 单 Audio，无 vision

        # Student forward
        if is_train:
            s_out = student(eeg, pcc)
        else:
            with torch.no_grad():
                s_out = student(eeg, pcc)

        losses = criterion(s_out, t_out, y, epoch)
        loss   = losses['loss']

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        n = eeg.size(0)
        for k in meters:
            meters[k].update(losses[k].item(), n)

        all_preds.extend(s_out['logits'].argmax(-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {**{k: m.avg for k, m in meters.items()}, 'acc': acc, 'f1': f1}


@torch.no_grad()
def evaluate_final(cfg, teacher, student, loader, criterion) -> dict:
    """测试集详细评估，额外返回 Teacher EEG 分支的参考精度和混淆矩阵"""
    student.eval(); teacher.eval()
    meters = {k: AverageMeter()
              for k in ('loss', 'l_ce', 'l_graph', 'l_temporal', 'l_logits')}
    s_preds, t_preds, all_labels = [], [], []

    for eeg, pcc, audio, y in loader:
        eeg, pcc, audio, y = (
            eeg.to(cfg.device), pcc.to(cfg.device),
            audio.to(cfg.device), y.to(cfg.device))

        t_out  = teacher(eeg, pcc, audio)
        s_out  = student(eeg, pcc)
        losses = criterion(s_out, t_out, y)

        n = eeg.size(0)
        for k in meters:
            meters[k].update(losses[k].item(), n)

        s_preds.extend(s_out['logits'].argmax(-1).cpu().numpy())
        t_preds.extend(t_out['eeg_logits'].argmax(-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return {
        **{k: m.avg for k, m in meters.items()},
        's_acc': accuracy_score(all_labels, s_preds),
        's_f1':  f1_score(all_labels, s_preds, average='weighted', zero_division=0),
        't_acc': accuracy_score(all_labels, t_preds),
        'cm':    confusion_matrix(all_labels, s_preds),
    }


# =============================================================================
# 7. 主程序
# =============================================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f'{cfg.exp_name}_best.pt')

    # ── 1. 数据 ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(cfg)

    # ── 2. 模型 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)

    student = ST_GCLSTM(
        num_nodes   = cfg.num_nodes,
        in_features = cfg.eeg_in_features,
        gcn_hidden  = cfg.gcn_hidden,
        gcn_out     = cfg.gcn_out,
        lstm_hidden = cfg.lstm_hidden,
        lstm_layers = cfg.lstm_layers,
        fc_hidden   = cfg.fc_hidden,
        num_classes = cfg.num_classes,
        dropout     = cfg.dropout,
    ).to(cfg.device)

    teacher = TeacherModel(
        audio_input_dim  = cfg.audio_input_dim,
        audio_hidden_dim = cfg.audio_hidden_dim,
        audio_num_layers = cfg.audio_num_layers,
        audio_dropout    = cfg.audio_dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.eeg_in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        dk               = cfg.dk,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(cfg.device)

    # 加载预训练 Teacher
    t_ckpt = os.path.join(cfg.save_dir, 'teacher_pme4_best.pt')
    if os.path.exists(t_ckpt):
        state = torch.load(t_ckpt, map_location=cfg.device, weights_only=True)
        teacher.load_state_dict(state['model_state'])
        print(f"✅ Teacher loaded from {t_ckpt}")
    else:
        print(f"⚠️  Teacher checkpoint not found: {t_ckpt}  (随机初始化)")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    s_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Student params : {s_params:,}")
    print(f"Teacher params : {t_params:,}  (frozen)")

    # ── 3. 损失 & 优化器 & 调度器 ────────────────────────────────────────────
    criterion = DistillationLoss(cfg)
    optimizer = optim.Adam(student.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    # ── 4. 训练循环 ───────────────────────────────────────────────────────────
    print(f"\n🚀 Distillation | "
          f"W_CE={cfg.base_w_ce}  W_Graph={cfg.base_w_graph}  "
          f"W_Temporal={cfg.base_w_temporal}  W_Logits={cfg.base_w_logits}  "
          f"Seed={cfg.seed}")
    print("-"*90)

    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = run_epoch(cfg, teacher, student, train_loader,
                       criterion, optimizer, epoch)
        vl = run_epoch(cfg, teacher, student, val_loader,
                       criterion, epoch=epoch)
        scheduler.step(vl['acc'])
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        w          = criterion.get_weights(epoch)

        print(f"Epoch {epoch:03d} | "
              f"Train Acc: {tr['acc']:.4f}  Val Acc: {vl['acc']:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")
        print(f"         Weights : CE={w['w_ce']:.2f}  Graph={w['w_graph']:.2f}  "
              f"Temporal={w['w_temporal']:.2f}  Logits={w['w_logits']:.2f}")
        print(f"         Train Loss: {tr['loss']:.4f}  "
              f"(CE={tr['l_ce']:.4f}  Graph={tr['l_graph']:.4f}  "
              f"Temporal={tr['l_temporal']:.4f}  Logits={tr['l_logits']:.4f})")
        print(f"         Val   Loss: {vl['loss']:.4f}  "
              f"(CE={vl['l_ce']:.4f}  Graph={vl['l_graph']:.4f}  "
              f"Temporal={vl['l_temporal']:.4f}  Logits={vl['l_logits']:.4f})")

        if vl['acc'] > best_val_acc:
            best_val_acc   = vl['acc']
            patience_count = 0
            torch.save({'model_state': student.state_dict(),
                        'val_acc': best_val_acc,
                        'val_f1':  vl['f1'],
                        'epoch':   epoch}, ckpt_path)
            print(f"  ✨ Best student saved  (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"⏹  Early stopping at epoch {epoch}")
                break

    # ── 5. 测试 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Loading best student from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
    student.load_state_dict(ckpt['model_state'])

    te = evaluate_final(cfg, teacher, student, test_loader, criterion)

    label_names = ['neg', 'pos']
    print(f"\n✅ Test Result (Student):")
    print(f"   Accuracy : {te['s_acc']:.4f}")
    print(f"   F1 Score : {te['s_f1']:.4f}")
    print(f"   Loss     : {te['loss']:.4f}  "
          f"(CE={te['l_ce']:.4f}  Graph={te['l_graph']:.4f}  "
          f"Temporal={te['l_temporal']:.4f}  Logits={te['l_logits']:.4f})")
    print(f"\n📊 Teacher EEG Branch (Reference):")
    print(f"   Accuracy : {te['t_acc']:.4f}")
    print(f"\n  Confusion Matrix ({' / '.join(label_names)}):")
    print(te['cm'])
    print("="*60)


if __name__ == '__main__':
    main()