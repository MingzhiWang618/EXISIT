"""
PME4 Teacher Training Script
Audio（OpenSMILE）指导 EEG 分类，单 Audio 模态替代原版 AV 双模态
"""

import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Optional

from PME4.dataset.dataset import create_pme4_dataloaders, CrossSubjectPME4Dataset
from PME4.model.Teacher   import TeacherModel, TeacherLoss


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root    = '/data2/zhiwen/dataset/PME4'
    normalize    = True
    num_classes  = 2

    # ── 模型 ──────────────────────────────────────────────────────────────────
    # Audio encoder
    audio_input_dim  = 25       # OpenSMILE eGeMAPS F=25
    audio_hidden_dim = 128
    audio_num_layers = 2
    audio_dropout    = 0.5

    # EEG / GCN / LSTM
    num_nodes       = 8         # PME4 电极数
    eeg_in_features = 5         # DE 5 频带
    gcn_hidden      = 64
    gcn_out         = 64
    lstm_hidden     = 64
    lstm_layers     = 1
    eeg_dropout     = 0.5
    dk              = 32
    fc_hidden       = 64

    # ── 损失权重 ──────────────────────────────────────────────────────────────
    w_audio = 0.4
    w_eeg   = 0.6

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 1e-3
    weight_decay = 1e-3
    patience     = 20

    # ── 输出 ──────────────────────────────────────────────────────────────────
    save_dir  = './checkpoints'
    exp_name  = 'teacher_pme4'

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 2024


# ═══════════════════════════════════════════════════════════════════════════════
# 多模态 Dataset（EEG + PCC + Audio）
# ═══════════════════════════════════════════════════════════════════════════════

class PME4MultiModalDataset(Dataset):
    """
    将 EEG/PCC 和 Audio 在 trial 维度对齐后合并。
    EEG   : [N_eeg, T, C, 5]
    PCC   : [N_eeg, T, C, C]
    Audio : [N_audio, T_a, F_a]
    Labels: 取 EEG 的 label（EEG 有缺失文件，以 EEG 为基准对齐）
    """
    def __init__(self,
                 eeg:   np.ndarray,
                 pcc:   np.ndarray,
                 audio: np.ndarray,
                 y:     np.ndarray):
        # PME4 EEG 和 Audio trial 数可能不一致（EEG 有 9 个缺失文件）
        # 以较少的那个为准（此处以 EEG 为基准，audio 截断对齐）
        n = min(len(eeg), len(audio), len(y))
        self.eeg   = torch.tensor(eeg[:n],   dtype=torch.float32)
        self.pcc   = torch.tensor(pcc[:n],   dtype=torch.float32)
        self.audio = torch.tensor(audio[:n], dtype=torch.float32)
        self.y     = torch.tensor(y[:n],     dtype=torch.long)
        print(f"  📦 MultiModal samples: {n}  "
              f"(eeg={len(eeg)}, audio={len(audio)}, y={len(y)})")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.eeg[idx], self.pcc[idx], self.audio[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def build_loaders(cfg: Config):
    """
    分别加载 EEG（+PCC）和 Audio，然后合并成 MultiModalDataset。
    两路数据独立归一化，保持与单模态训练的一致性。
    """
    manager = CrossSubjectPME4Dataset(root=cfg.data_root, fs=1000)

    # ── EEG + PCC ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading EEG + PCC ...")
    (tr_eeg, tr_ey, tr_pcc), (va_eeg, va_ey, va_pcc), (te_eeg, te_ey, te_pcc) = \
        manager.get_all_splits(
            segment_1s=True, extract_de=True,
            normalize=True, compute_pcc=True)

    # ── Audio（OpenSMILE）─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading Audio (OpenSMILE) ...")
    (tr_aud, tr_ay), (va_aud, va_ay), (te_aud, te_ay) = \
        manager.get_all_splits(
            modality="audio_opensmile",
            normalize=True)

    # ── 合并 ──────────────────────────────────────────────────────────────────
    def make_loader(eeg, pcc, aud, y, shuffle):
        ds = PME4MultiModalDataset(eeg, pcc, aud, y)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    return (make_loader(tr_eeg, tr_pcc, tr_aud, tr_ey, shuffle=True),
            make_loader(va_eeg, va_pcc, va_aud, va_ey, shuffle=False),
            make_loader(te_eeg, te_pcc, te_aud, te_ey, shuffle=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 单 epoch 训练 / 评估
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter()
    audio_preds, eeg_preds, all_labels = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for eeg, pcc, audio, labels in loader:
            eeg, pcc, audio, labels = (
                eeg.to(device), pcc.to(device),
                audio.to(device), labels.to(device))

            out    = model(eeg, pcc, audio)
            losses = criterion(out, labels)
            loss   = losses['loss']

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_meter.update(loss.item(), labels.size(0))
            audio_preds.extend(out['audio_logits'].argmax(1).cpu().numpy())
            eeg_preds.extend(  out['eeg_logits'].argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    audio_acc = accuracy_score(all_labels, audio_preds)
    eeg_acc   = accuracy_score(all_labels, eeg_preds)
    eeg_f1    = f1_score(all_labels, eeg_preds, average='weighted', zero_division=0)

    return {
        'loss':      loss_meter.avg,
        'audio_acc': audio_acc,
        'eeg_acc':   eeg_acc,
        'eeg_f1':    eeg_f1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练流程
# ═══════════════════════════════════════════════════════════════════════════════

def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f'{cfg.exp_name}_best.pt')
    device    = torch.device(cfg.device)

    # ── 1. 数据 ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_loaders(cfg)

    # ── 2. 模型 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    model = TeacherModel(
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
        eeg_dropout      = cfg.eeg_dropout,
        dk               = cfg.dk,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)

    print(f"Model  : TeacherModel (Audio → EEG)")
    print(f"Device : {device}")
    print(f"Params : {count_parameters(model):,}")

    # ── 3. 损失 & 优化器 & 调度器 ────────────────────────────────────────────
    criterion = TeacherLoss(w_audio=cfg.w_audio, w_eeg=cfg.w_eeg)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    # ── 4. 训练循环 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Epoch':>6} | {'LR':>8} | "
          f"{'Tr-Loss':>8} {'Tr-Audio':>9} {'Tr-EEG':>7} {'Tr-F1':>7} | "
          f"{'Va-Loss':>8} {'Va-Audio':>9} {'Va-EEG':>7} {'Va-F1':>7}")
    print("-"*100)

    best_val_eeg_acc = 0.0
    patience_count   = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, optimizer,
                       device, is_train=True)
        vl = run_epoch(model, val_loader,   criterion, optimizer,
                       device, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {lr_now:>8.2e} | "
              f"{tr['loss']:>8.4f} {tr['audio_acc']:>9.4f} "
              f"{tr['eeg_acc']:>7.4f} {tr['eeg_f1']:>7.4f} | "
              f"{vl['loss']:>8.4f} {vl['audio_acc']:>9.4f} "
              f"{vl['eeg_acc']:>7.4f} {vl['eeg_f1']:>7.4f}"
              f"  {elapsed:.1f}s", end="")

        if vl['eeg_acc'] > best_val_eeg_acc:
            best_val_eeg_acc = vl['eeg_acc']
            patience_count   = 0
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_eeg_acc':     best_val_eeg_acc,
                'val_audio_acc':   vl['audio_acc'],
                'val_eeg_f1':      vl['eeg_f1'],
            }, ckpt_path)
            print("  ✅")
        else:
            patience_count += 1
            print(f"  (patience {patience_count}/{cfg.patience})")
            if patience_count >= cfg.patience:
                print(f"\n⏹  Early stopping at epoch {epoch}")
                break

    # ── 5. 测试 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Best val EEG acc: {best_val_eeg_acc:.4f}  →  loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])

    te = run_epoch(model, test_loader, criterion, None,
                   device, is_train=False)

    # 详细混淆矩阵
    model.eval()
    all_eeg_preds, all_labels = [], []
    with torch.no_grad():
        for eeg, pcc, audio, labels in test_loader:
            out = model(eeg.to(device), pcc.to(device), audio.to(device))
            all_eeg_preds.extend(out['eeg_logits'].argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    label_names = ['neg', 'pos']
    cm = confusion_matrix(all_labels, all_eeg_preds)

    print(f"\n  Test Audio Acc : {te['audio_acc']:.4f}")
    print(f"  Test EEG   Acc : {te['eeg_acc']:.4f}")
    print(f"  Test EEG   F1  : {te['eeg_f1']:.4f}")
    print(f"  Test Loss      : {te['loss']:.4f}")
    print(f"\n  Confusion Matrix ({' / '.join(label_names)}):")
    print(cm)
    print("="*60)

    return te


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    cfg = Config()
    train(cfg)