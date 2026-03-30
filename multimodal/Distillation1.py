# train_student.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from multimodal.model.Teacher1 import TeacherModel
from multimodal.model.Student1  import StudentModel
from train_teacher1 import (
    MultiModalDataset, build_loaders,
    AUDIO_DIM, VISION_DIM, NUM_CLASSES, BATCH_SIZE,
    IN_FEATURES, LSTM_HIDDEN, LSTM_LAYERS,
    FC_HIDDEN, DROPOUT, AV_HIDDEN, DK, DEVICE,
    GCN_DIM,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

TEACHER_CKPT = './checkpoints/best_teacher1.pth'
SAVE_DIR     = './checkpoints'
EPOCHS       = 150
LR           = 1e-4
WEIGHT_DECAY = 1e-3
PATIENCE     = 20

W_CE   = 0.5
W_S    = 0.0   # S 矩阵 KL 对齐
W_ATTN = 10   # 时序注意力 KL 对齐


# ═══════════════════════════════════════════════════════════════════════════════
# 蒸馏损失
# ═══════════════════════════════════════════════════════════════════════════════

class DistillLoss(nn.Module):
    def __init__(self, w_ce=1.0, w_S=0.2, w_attn=0.2):
        super().__init__()
        self.w_ce   = w_ce
        self.w_S    = w_S
        self.w_attn = w_attn
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, s_out: dict, t_out: dict,
                labels: torch.Tensor) -> dict:
        # 分类损失
        l_ce = self.ce(s_out['logits'], labels)

        # S 矩阵对齐：[B,T,7,7] → [-1,7]，每行是 softmax 后的概率分布
        l_S = self.kl(
            s_out['R_inter'].reshape(-1, 7).log(),
            t_out['R_inter'].reshape(-1, 7).detach()
        )

        # 时序注意力对齐：[B,T]，都是 softmax 后的概率分布
        l_attn = self.kl(
            s_out['attn_t'].log(),
            t_out['attn_weights'].detach()
        )

        total = (self.w_ce   * l_ce
               + self.w_S    * l_S
               + self.w_attn * l_attn)

        return {
            'loss'  : total,
            'l_ce'  : l_ce,
            'l_S'   : l_S,
            'l_attn': l_attn,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Train / Eval
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(student, teacher, loader, criterion, optimizer, device):
    student.train()
    teacher.eval()

    meters = {k: AverageMeter() for k in
              ['loss', 'l_ce', 'l_S', 'l_attn']}
    all_preds, all_labels = [], []

    for eeg, audio, vision, labels in tqdm(loader, leave=False, desc='Train'):
        eeg, audio, vision, labels = (
            eeg.to(device), audio.to(device),
            vision.to(device), labels.to(device)
        )

        with torch.no_grad():
            t_out = teacher(eeg, audio, vision)

        s_out     = student(eeg)
        loss_dict = criterion(s_out, t_out, labels)
        loss      = loss_dict['loss']

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        B = labels.size(0)
        for k, v in loss_dict.items():
            meters[k].update(v.item(), B)

        all_preds.extend(s_out['logits'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1' : f1_score(all_labels, all_preds,
                        average='weighted', zero_division=0),
        **{k: m.avg for k, m in meters.items()}
    }


@torch.no_grad()
def evaluate(student, teacher, loader, criterion, device, desc='Val'):
    student.eval()
    teacher.eval()

    meters = {k: AverageMeter() for k in
              ['loss', 'l_ce', 'l_S', 'l_attn']}
    all_preds, all_labels = [], []

    for eeg, audio, vision, labels in tqdm(loader, leave=False, desc=desc):
        eeg, audio, vision, labels = (
            eeg.to(device), audio.to(device),
            vision.to(device), labels.to(device)
        )

        t_out     = teacher(eeg, audio, vision)
        s_out     = student(eeg)
        loss_dict = criterion(s_out, t_out, labels)

        B = labels.size(0)
        for k, v in loss_dict.items():
            meters[k].update(v.item(), B)

        all_preds.extend(s_out['logits'].argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1' : f1_score(all_labels, all_preds,
                        average='weighted', zero_division=0),
        'cm' : confusion_matrix(all_labels, all_preds),
        **{k: m.avg for k, m in meters.items()}
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    set_seed()
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader = build_loaders()

    # ── Teacher（加载并冻结） ─────────────────────────────────────────────────
    teacher = TeacherModel(
        audio_input_dim  = AUDIO_DIM,
        vision_input_dim = VISION_DIM,
        av_hidden_dim    = AV_HIDDEN,
        av_num_layers    = LSTM_LAYERS,
        av_dropout       = DROPOUT,
        eeg_in_features  = IN_FEATURES,
        gcn_dim          = GCN_DIM,
        lstm_hidden      = LSTM_HIDDEN,
        lstm_layers      = LSTM_LAYERS,
        eeg_dropout      = DROPOUT,
        dk               = DK,
        fc_hidden        = FC_HIDDEN,
        num_classes      = NUM_CLASSES,
    ).to(device)

    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded  (best val acc: {ckpt['best_eeg_acc']:.4f})")

    # ── Student ───────────────────────────────────────────────────────────────
    student = StudentModel(
        eeg_in_features = IN_FEATURES,
        gcn_dim         = GCN_DIM,
        lstm_hidden     = LSTM_HIDDEN,
        lstm_layers     = LSTM_LAYERS,
        fc_hidden       = FC_HIDDEN,
        num_classes     = NUM_CLASSES,
        dropout         = DROPOUT,
    ).to(device)

    s_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Student params : {s_params:,}")
    print(f"Teacher params : {t_params:,}")

    # ── 损失 & 优化器 ─────────────────────────────────────────────────────────
    criterion = DistillLoss(w_ce=W_CE, w_S=W_S, w_attn=W_ATTN)
    optimizer = optim.AdamW(student.parameters(),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_acc       = 0.0
    patience_count = 0
    ckpt_path      = os.path.join(SAVE_DIR, 'best_student.pth')

    print(f"\n{'='*90}")
    print(f"  {'Ep':>4}  {'LR':>8}  "
          f"{'Tr-Loss':>8} {'Tr-Acc':>7}  "
          f"{'Va-Loss':>8} {'Va-Acc':>7}  "
          f"{'l_ce':>7} {'l_S':>7} {'l_attn':>7}")
    print(f"{'='*90}")

    for epoch in range(1, EPOCHS + 1):

        tr = train_one_epoch(student, teacher, train_loader,
                             criterion, optimizer, device)
        va = evaluate(student, teacher, val_loader,
                      criterion, device, desc='Val')
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if va['acc'] > best_acc:
            best_acc       = va['acc']
            patience_count = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': student.state_dict(),
                'best_acc'   : best_acc,
            }, ckpt_path)
            flag = '  ✅'
        else:
            patience_count += 1
            flag = f'  ({patience_count}/{PATIENCE})'

        print(f"  {epoch:4d}  {lr_now:8.2e}  "
              f"{tr['loss']:8.4f} {tr['acc']:7.4f}  "
              f"{va['loss']:8.4f} {va['acc']:7.4f}  "
              f"{va['l_ce']:7.4f} {va['l_S']:7.4f} {va['l_attn']:7.4f}"
              f"{flag}")

        if patience_count >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print(f"\nBest val acc: {best_acc:.4f}  →  {ckpt_path}")
    student.load_state_dict(
        torch.load(ckpt_path, map_location=device)['model_state']
    )
    te = evaluate(student, teacher, test_loader,
                  criterion, device, desc='Test')

    print(f"\n{'='*40}")
    print(f"  Test Acc  : {te['acc']:.4f}")
    print(f"  Test F1   : {te['f1']:.4f}")
    print(f"  Test Loss : {te['loss']:.4f}")
    print(f"  l_ce      : {te['l_ce']:.4f}")
    print(f"  l_S       : {te['l_S']:.4f}")
    print(f"  l_attn    : {te['l_attn']:.4f}")
    print(f"{'='*40}")
    print("Confusion Matrix:")
    print(te['cm'])


if __name__ == '__main__':
    main()