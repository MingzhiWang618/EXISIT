import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from KDbaseline.model.Teacher import TeacherModel
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

    # ── 模型 ──────────────────────────────────────────────────────────────────
    num_nodes   = 30
    in_features = 5
    gcn_hidden  = 64
    gcn_out     = 64
    lstm_hidden = 64
    lstm_layers = 1
    fc_hidden   = 64
    num_classes = 5
    dropout     = 0.5

    audio_dim   = 25
    vision_dim  = 161
    av_hidden   = 64

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 5e-5
    weight_decay = 1e-3
    patience     = 20
    seed         = 2024
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── KD 超参 ───────────────────────────────────────────────────────────────
    temperature = 4.0   # 软标签温度，常用 2~6
    w_ce        = 1.0   # hard label loss 权重
    w_kd        = 1.0   # soft logits KL 权重

    # ── 路径 ──────────────────────────────────────────────────────────────────
    save_dir    = './checkpoints'
    teacher_ckpt = './checkpoints/best_teacher.pth'
    student_ckpt = './checkpoints/best_student_kd.pth'


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


# =============================================================================
# KD Loss
# =============================================================================

class KDLoss(nn.Module):
    """
    L = w_ce * CE(s_logits, labels)
      + w_kd * T^2 * KL(softmax(s/T) || softmax(t/T))
    """
    def __init__(self, temperature: float = 4.0,
                 w_ce: float = 1.0, w_kd: float = 1.0):
        super().__init__()
        self.T    = temperature
        self.w_ce = w_ce
        self.w_kd = w_kd
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, labels):
        l_ce = self.ce(s_logits, labels)

        s_log = F.log_softmax(s_logits / self.T, dim=-1)
        t_soft = F.softmax(t_logits.detach() / self.T, dim=-1)
        l_kd  = F.kl_div(s_log, t_soft, reduction='batchmean') * (self.T ** 2)

        total = self.w_ce * l_ce + self.w_kd * l_kd
        return {'loss': total, 'l_ce': l_ce, 'l_kd': l_kd}


# =============================================================================
# 工具
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# =============================================================================
# 单 epoch
# =============================================================================

def run_epoch(cfg, teacher, student, loader, criterion, optimizer=None, epoch=0):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()
    teacher.eval()

    meters = {k: AverageMeter() for k in ('loss', 'l_ce', 'l_kd')}
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for eeg, pcc, audio, vision, y in loader:
            eeg, pcc, audio, vision, y = [d.to(cfg.device)
                                          for d in (eeg, pcc, audio, vision, y)]

            with torch.no_grad():
                t_out = teacher(eeg, pcc, audio, vision)   # teacher: {logits, ...}

            if is_train:
                s_out = student(eeg, pcc)
            else:
                with torch.no_grad():
                    s_out = student(eeg, pcc)

            losses = criterion(s_out['logits'], t_out['logits'], y)

            if is_train:
                optimizer.zero_grad()
                losses['loss'].backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            n = eeg.size(0)
            for k in meters:
                meters[k].update(losses[k].item(), n)

            all_preds.extend(s_out['logits'].argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'acc'  : acc,
        'f1'   : f1,
        'loss' : meters['loss'].avg,
        'l_ce' : meters['l_ce'].avg,
        'l_kd' : meters['l_kd'].avg,
    }


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
    manager = CrossSubjectMultiModalDataset(
        cfg.data_root_dict,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )

    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)

    (tr_aud, _), (va_aud, _), (te_aud, _) = \
        manager.get_all_splits('audio', normalize=True)

    (tr_vis, _), (va_vis, _), (te_vis, _) = \
        manager.get_all_splits('vision', normalize=True)

    def make_loader(eeg, pcc, y, aud, vis, shuffle):
        ds = AlignedDistillDataset(eeg, pcc, y, aud, vis)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = make_loader(tr_eeg, tr_pcc, tr_y, tr_aud, tr_vis, True)
    val_loader   = make_loader(va_eeg, va_pcc, va_y, va_aud, va_vis, False)
    test_loader  = make_loader(te_eeg, te_pcc, te_y, te_aud, te_vis, False)

    # ── 模型 ──────────────────────────────────────────────────────────────────
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

    teacher = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden,
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

    if not os.path.exists(cfg.teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found: {cfg.teacher_ckpt}")
    teacher.load_state_dict(
        torch.load(cfg.teacher_ckpt, map_location=device)['model_state'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student params : {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")

    # ── 损失 / 优化器 ────────────────────────────────────────────────────────
    criterion = KDLoss(temperature=cfg.temperature,
                       w_ce=cfg.w_ce, w_kd=cfg.w_kd)
    optimizer = optim.Adam(student.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=7)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0

    print(f"\n🚀 KD  T={cfg.temperature}  w_ce={cfg.w_ce}  w_kd={cfg.w_kd}  seed={cfg.seed}")
    print(f"{'='*75}")
    print(f"  {'Ep':>4}  {'LR':>8}  "
          f"{'Tr-Loss':>8} {'Tr-CE':>7} {'Tr-KD':>7} {'Tr-Acc':>7}  "
          f"{'Va-Loss':>8} {'Va-Acc':>7}")
    print(f"{'='*75}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr = run_epoch(cfg, teacher, student, train_loader,
                       criterion, optimizer, epoch)
        va = run_epoch(cfg, teacher, student, val_loader,
                       criterion, epoch=epoch)

        scheduler.step(va['acc'])
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        flag = ''
        if va['acc'] > best_val_acc:
            best_val_acc   = va['acc']
            patience_count = 0
            torch.save({'epoch'      : epoch,
                        'model_state': student.state_dict(),
                        'val_acc'    : best_val_acc}, cfg.student_ckpt)
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

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print(f"\n🔍 Best val acc: {best_val_acc:.4f}  →  {cfg.student_ckpt}")
    student.load_state_dict(
        torch.load(cfg.student_ckpt, map_location=device)['model_state'])
    student.eval()

    # 测试集指标
    te = run_epoch(cfg, teacher, student, test_loader, criterion, epoch=0)

    # teacher 在测试集上的参考精度
    t_preds, t_labels = [], []
    with torch.no_grad():
        for eeg, pcc, audio, vision, y in test_loader:
            eeg, pcc, audio, vision, y = [d.to(device)
                                          for d in (eeg, pcc, audio, vision, y)]
            t_out = teacher(eeg, pcc, audio, vision)
            t_preds.extend(t_out['logits'].argmax(1).cpu().numpy())
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