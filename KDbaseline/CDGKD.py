# train_cdgkd.py
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from KDbaseline.model.Teacher import TeacherModel   # 复用作 TA
from KDbaseline.model.Student import ST_GCLSTM
from dataset.dataset import CrossSubjectMultiModalDataset


# =============================================================================
# Config
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root_dict = {
        'eeg'   : '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'audio' : '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace',
    }

    # ── 模型（EEG 分支，Teacher / TA / Student 共享）─────────────────────────
    num_nodes   = 30
    in_features = 5
    gcn_hidden  = 64
    gcn_out     = 64
    lstm_hidden = 64
    lstm_layers = 1
    fc_hidden   = 64
    num_classes = 5
    dropout     = 0.5

    # ── AV 分支容量逐级缩减（Teacher → TA1 → TA2）───────────────────────────
    audio_dim      = 25
    vision_dim     = 161
    av_hidden_teacher = 64    # Teacher：完整 AV
    av_hidden_ta1     = 32    # TA1：减半
    av_hidden_ta2     = 16    # TA2：再减半

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 1e-3
    weight_decay = 1e-3
    patience     = 20
    seed         = 2024
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── CDGKD 超参（论文 β=0.5）──────────────────────────────────────────────
    beta        = 0.5    # CE 与 KD loss 的权重
    temperature = 4.0    # 软标签温度

    # ── 路径 ──────────────────────────────────────────────────────────────────
    save_dir      = './checkpoints'
    teacher_ckpt  = './checkpoints/best_teacher_proto.pth'
    ta1_ckpt      = './checkpoints/best_ta1.pth'
    ta2_ckpt      = './checkpoints/best_ta2.pth'
    student_ckpt  = './checkpoints/best_student_cdgkd.pth'


# =============================================================================
# Dataset
# =============================================================================

class AlignedDistillDataset(Dataset):
    def __init__(self, eeg, pcc, labels, audio, vision):
        self.eeg    = eeg
        self.pcc    = pcc
        self.labels = labels
        self.audio  = audio
        self.vision = vision

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.eeg[idx], self.pcc[idx],
                self.audio[idx], self.vision[idx],
                self.labels[idx])


def build_loaders(cfg):
    manager = CrossSubjectMultiModalDataset(
        cfg.data_root_dict,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )
    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True,
                               normalize=True, compute_pcc=True)
    (tr_aud, _), (va_aud, _), (te_aud, _) = \
        manager.get_all_splits('audio', normalize=True)
    (tr_vis, _), (va_vis, _), (te_vis, _) = \
        manager.get_all_splits('vision', normalize=True)

    def make(eeg, pcc, y, aud, vis, shuffle):
        ds = AlignedDistillDataset(eeg, pcc, y, aud, vis)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    return (make(tr_eeg, tr_pcc, tr_y, tr_aud, tr_vis, True),
            make(va_eeg, va_pcc, va_y, va_aud, va_vis, False),
            make(te_eeg, te_pcc, te_y, te_aud, te_vis, False))


# =============================================================================
# CDGKD Loss（论文公式 9 + 随机学习策略）
# =============================================================================

class CDGKDLoss(nn.Module):
    """
    L = (1-β)·L_CE(curr_logits, labels)
      + β · (1/|S_i|) · Σ_{j∈S_i} KL(curr_logits/T ‖ upper_logits_j/T) · T²

    随机学习策略：从上级网络 logits 列表中随机采样子集 S_i。
    训练时 stochastic=True，验证/测试时 stochastic=False（用全部上级）。
    """
    def __init__(self, beta: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.beta = beta
        self.T    = temperature
        self.ce   = nn.CrossEntropyLoss()

    def forward(self,
                curr_logits: torch.Tensor,
                upper_logits_list: list,
                labels: torch.Tensor,
                stochastic: bool = True) -> dict:
        """
        curr_logits       : [B, C]  当前网络（TA_i 或 Student）的 logits
        upper_logits_list : list of [B, C]，所有上级网络的 logits
        labels            : [B]
        stochastic        : 是否随机采样子集（训练时 True，验证时 False）
        """
        l_ce = self.ce(curr_logits, labels)

        # 随机学习策略：从上级 logits 中随机采样非空子集
        if stochastic and len(upper_logits_list) > 1:
            k      = random.randint(1, len(upper_logits_list))
            subset = random.sample(upper_logits_list, k)
        else:
            subset = upper_logits_list

        # KL divergence 对每个上级求平均
        kl_losses = []
        s_log = F.log_softmax(curr_logits / self.T, dim=-1)
        for t_logits in subset:
            t_soft = F.softmax(t_logits.detach() / self.T, dim=-1)
            kl     = F.kl_div(s_log, t_soft, reduction='batchmean') * (self.T ** 2)
            kl_losses.append(kl)

        l_kd = torch.stack(kl_losses).mean()
        loss = (1 - self.beta) * l_ce + self.beta * l_kd

        return {'loss': loss, 'l_ce': l_ce, 'l_kd': l_kd}


# =============================================================================
# 工具
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    acc = ckpt.get('best_val_acc', ckpt.get('val_acc', '?'))
    print(f"  Loaded {ckpt_path}  (val_acc={acc})")
    return model


# =============================================================================
# 单 epoch：TA 或 Student 训练/验证
# =============================================================================

def run_epoch(cfg, current_net, upper_nets, loader,
              criterion, optimizer=None, epoch=0,
              current_uses_av=True):
    """
    current_net     : 当前训练的网络（TA 或 Student）
    upper_nets      : list of 已固定的上级网络（Teacher, TA1, TA2, ...）
    current_uses_av : 当前网络是否需要 audio/vision 输入（Student 不需要）
    """
    is_train   = optimizer is not None
    stochastic = is_train     # 只在训练时启用随机学习策略
    current_net.train() if is_train else current_net.eval()

    meters = {k: AverageMeter() for k in ('loss', 'l_ce', 'l_kd')}
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for eeg, pcc, audio, vision, y in loader:
            eeg, pcc, audio, vision, y = [
                d.to(cfg.device) for d in (eeg, pcc, audio, vision, y)]

            # ── 上级网络推理（全部冻结，无梯度）────────────────────────────
            upper_logits_list = []
            with torch.no_grad():
                for net in upper_nets:
                    # upper_nets 全部是 TeacherModel 结构（有 AV 输入）
                    out = net(eeg, pcc, audio, vision)
                    upper_logits_list.append(out['logits'])

            # ── 当前网络前向 ────────────────────────────────────────────────
            if current_uses_av:
                # TA：TeacherModel 结构，需要 AV 输入
                if is_train:
                    curr_out = current_net(eeg, pcc, audio, vision)
                else:
                    with torch.no_grad():
                        curr_out = current_net(eeg, pcc, audio, vision)
            else:
                # Student：ST_GCLSTM，只用 EEG
                if is_train:
                    curr_out = current_net(eeg, pcc)
                else:
                    with torch.no_grad():
                        curr_out = current_net(eeg, pcc)

            curr_logits = curr_out['logits']

            # ── 损失计算 ────────────────────────────────────────────────────
            losses = criterion(curr_logits, upper_logits_list, y, stochastic)

            if is_train:
                optimizer.zero_grad()
                losses['loss'].backward()
                nn.utils.clip_grad_norm_(current_net.parameters(), 1.0)
                optimizer.step()

            n = eeg.size(0)
            for k in meters:
                meters[k].update(losses[k].item(), n)
            all_preds.extend(curr_logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'acc' : acc, 'f1': f1,
        'loss': meters['loss'].avg,
        'l_ce': meters['l_ce'].avg,
        'l_kd': meters['l_kd'].avg,
    }


# =============================================================================
# 单个网络的完整训练流程（含 early stopping）
# =============================================================================

def train_one_level(cfg, name, current_net, upper_nets,
                    train_loader, val_loader,
                    criterion, ckpt_path,
                    current_uses_av=True):

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, current_net.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    best_val_acc   = 0.0
    patience_count = 0

    print(f"\n{'='*70}")
    print(f"  Training {name}   upper={[type(n).__name__ for n in upper_nets]}")
    print(f"{'='*70}")
    print(f"  {'Ep':>4}  {'LR':>8}  "
          f"{'Tr-Loss':>8} {'Tr-CE':>7} {'Tr-KD':>7} {'Tr-Acc':>7}  "
          f"{'Va-Loss':>8} {'Va-Acc':>7}")
    print(f"{'='*70}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr = run_epoch(cfg, current_net, upper_nets, train_loader,
                       criterion, optimizer, epoch, current_uses_av)
        va = run_epoch(cfg, current_net, upper_nets, val_loader,
                       criterion, epoch=epoch,
                       current_uses_av=current_uses_av)

        scheduler.step(va['acc'])
        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        flag = ''
        if va['acc'] > best_val_acc:
            best_val_acc   = va['acc']
            patience_count = 0
            torch.save({'epoch'       : epoch,
                        'model_state' : current_net.state_dict(),
                        'best_val_acc': best_val_acc}, ckpt_path)
            flag = '  ✅'
        else:
            patience_count += 1
            flag = f'  ({patience_count}/{cfg.patience})'

        print(f"  {epoch:4d}  {lr_now:8.2e}  "
              f"{tr['loss']:8.4f} {tr['l_ce']:7.4f} {tr['l_kd']:7.4f} {tr['acc']:7.4f}  "
              f"{va['loss']:8.4f} {va['acc']:7.4f}"
              f"{flag}  [{elapsed:.1f}s]")

        if patience_count >= cfg.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

    print(f"\n  Best val acc ({name}): {best_val_acc:.4f}  →  {ckpt_path}")
    # 加载最优权重供下一级使用
    current_net.load_state_dict(
        torch.load(ckpt_path, map_location=cfg.device)['model_state'])
    return current_net, best_val_acc


# =============================================================================
# 主程序
# =============================================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # ── 数据 ──────────────────────────────────────────────────────────────────
    print("\n📦 Loading data ...")
    train_loader, val_loader, test_loader = build_loaders(cfg)

    # ── CDGKD Loss ────────────────────────────────────────────────────────────
    criterion = CDGKDLoss(beta=cfg.beta, temperature=cfg.temperature)

    # =========================================================================
    # Step 0：加载预训练 Teacher（固定，全程不更新）
    # =========================================================================
    print("\n📥 Loading pretrained Teacher ...")
    teacher = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden_teacher,
        av_num_layers    = cfg.lstm_layers,
        av_dropout       = cfg.dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)
    load_model(teacher, cfg.teacher_ckpt, device)
    freeze(teacher)

    # =========================================================================
    # Step 1：训练 TA1  ← {Teacher}
    # 论文公式 7：A1 ← {T}
    # =========================================================================
    ta1 = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden_ta1,   # 32，AV 减半
        av_num_layers    = cfg.lstm_layers,
        av_dropout       = cfg.dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)

    ta1, _ = train_one_level(
        cfg, name='TA1',
        current_net   = ta1,
        upper_nets    = [teacher],           # 论文公式 7：A1 ← {T}
        train_loader  = train_loader,
        val_loader    = val_loader,
        criterion     = criterion,
        ckpt_path     = cfg.ta1_ckpt,
        current_uses_av = True,
    )
    freeze(ta1)

    # =========================================================================
    # Step 2：训练 TA2  ← {Teacher, TA1}
    # 论文公式 7：A2 ← {T, A1}，随机采样子集
    # =========================================================================
    ta2 = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden_ta2,   # 16，再减半
        av_num_layers    = cfg.lstm_layers,
        av_dropout       = cfg.dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)

    ta2, _ = train_one_level(
        cfg, name='TA2',
        current_net   = ta2,
        upper_nets    = [teacher, ta1],      # 论文公式 7：A2 ← {T, A1}
        train_loader  = train_loader,
        val_loader    = val_loader,
        criterion     = criterion,
        ckpt_path     = cfg.ta2_ckpt,
        current_uses_av = True,
    )
    freeze(ta2)

    # =========================================================================
    # Step 3：训练 Student  ← {Teacher, TA1, TA2}
    # 论文公式 8：S ← {T} ∪ A，随机采样子集
    # =========================================================================
    student = ST_GCLSTM(
        num_nodes   = cfg.num_nodes,
        in_features = cfg.in_features,
        gcn_hidden  = cfg.gcn_hidden,
        gcn_out     = cfg.gcn_out,
        lstm_hidden = cfg.lstm_hidden,
        lstm_layers = cfg.lstm_layers,
        fc_hidden   = cfg.fc_hidden,
        num_classes = cfg.num_classes,
        dropout     = cfg.dropout,
    ).to(device)

    student, best_student_acc = train_one_level(
        cfg, name='Student',
        current_net   = student,
        upper_nets    = [teacher, ta1, ta2],  # 论文公式 8：S ← {T} ∪ A
        train_loader  = train_loader,
        val_loader    = val_loader,
        criterion     = criterion,
        ckpt_path     = cfg.student_ckpt,
        current_uses_av = False,              # Student 只用 EEG
    )

    # =========================================================================
    # 测试
    # =========================================================================
    print(f"\n🔍 Best val acc (Student): {best_student_acc:.4f}")
    student.load_state_dict(
        torch.load(cfg.student_ckpt, map_location=device)['model_state'])
    student.eval()

    te = run_epoch(cfg, student, [teacher, ta1, ta2],
                   test_loader, criterion,
                   epoch=0, current_uses_av=False)

    # Teacher 测试参考精度
    t_preds, t_labels = [], []
    with torch.no_grad():
        for eeg, pcc, audio, vision, y in test_loader:
            eeg, pcc, audio, vision, y = [
                d.to(device) for d in (eeg, pcc, audio, vision, y)]
            out = teacher(eeg, pcc, audio, vision)
            t_preds.extend(out['logits'].argmax(1).cpu().numpy())
            t_labels.extend(y.cpu().numpy())
    t_acc = accuracy_score(t_labels, t_preds)

    print(f"\n{'='*45}")
    print(f"  Student Test Acc  : {te['acc']:.4f}")
    print(f"  Student Test F1   : {te['f1']:.4f}")
    print(f"  Student Test Loss : {te['loss']:.4f}  "
          f"(CE {te['l_ce']:.4f} | KD {te['l_kd']:.4f})")
    print(f"  Teacher Test Acc  : {t_acc:.4f}  (reference)")
    print(f"{'='*45}\n")


if __name__ == '__main__':
    main()