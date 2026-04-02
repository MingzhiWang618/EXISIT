import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import f1_score
from dataset.dataset import CrossSubjectMultiModalDataset
from KDbaseline.model.Teacher import TeacherModel
from KDbaseline.model.Student import ST_GCLSTM


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DATA_ROOT = {
    'eeg'   : '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
    'audio' : '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw',
    'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace',
}

AUDIO_DIM    = 25
VISION_DIM   = 161
NUM_CLASSES  = 5
BATCH_SIZE   = 64
LR           = 5e-5
WEIGHT_DECAY = 1e-4

NUM_NODES    = 30
IN_FEATURES  = 5
GCN_HIDDEN   = 64
GCN_OUT      = 64
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 1
FC_HIDDEN    = 64
DROPOUT      = 0.5
AV_HIDDEN    = 64

EPOCHS_STAGE1 = 50
EPOCHS_STAGE2 = 150
PATIENCE      = 20
TEMPERATURE   = 4.0
SAVE_DIR      = './checkpoints'
DEVICE        = 'cuda'
TEACHER_CKPT  = './checkpoints/best_teacher.pth'

# Teacher fused 维度 = eeg_dim + av_dim = LSTM_HIDDEN*2 + AV_HIDDEN*2*2
# eeg_dim  = LSTM_HIDDEN * 2          = 64 * 2 = 128
# av_dim   = AV_HIDDEN*2 (audio) + AV_HIDDEN*2 (vision) = 128 + 128 = 256
STUDENT_DIM = LSTM_HIDDEN * 2                    # 128
TEACHER_DIM = LSTM_HIDDEN * 2 + AV_HIDDEN * 4   # 128 + 256 = 384


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalDataset(Dataset):
    def __init__(self, eeg, pcc, audio, vision, labels, vision_max_len=None):
        self.eeg   = torch.tensor(eeg,    dtype=torch.float32)
        self.pcc   = torch.tensor(pcc,    dtype=torch.float32)
        self.audio = torch.tensor(audio,  dtype=torch.float32)
        self.y     = torch.tensor(labels, dtype=torch.long)

        if vision_max_len is None:
            vision_max_len = max(s.shape[0] for s in vision)
        F_v = vision[0].shape[-1]
        arr = np.zeros((len(vision), vision_max_len, F_v), dtype=np.float32)
        for i, s in enumerate(vision):
            t = min(s.shape[0], vision_max_len)
            arr[i, :t] = s[:t]
        self.vision = torch.tensor(arr, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.eeg[idx], self.pcc[idx],
                self.audio[idx], self.vision[idx], self.y[idx])


def build_loaders(vision_max_len=None):
    manager = CrossSubjectMultiModalDataset(
        DATA_ROOT,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )
    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)
    (tr_aud, _), (va_aud, _), (te_aud, _) = \
        manager.get_all_splits('audio', extract_de=False, normalize=True)
    (tr_vis, _), (va_vis, _), (te_vis, _) = \
        manager.get_all_splits('vision', extract_de=False, normalize=True)

    def make(eeg, pcc, aud, vis, y, shuffle):
        ds = MultiModalDataset(eeg, pcc, aud, vis, y, vision_max_len)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    return (make(tr_eeg, tr_pcc, tr_aud, tr_vis, tr_y, True),
            make(va_eeg, va_pcc, va_aud, va_vis, va_y, False),
            make(te_eeg, te_pcc, te_aud, te_vis, te_y, False))


# ═══════════════════════════════════════════════════════════════════════════════
# HintRegressor：student eeg_feat → teacher fused 维度
# ═══════════════════════════════════════════════════════════════════════════════

class HintRegressor(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = (
            nn.Linear(student_dim, teacher_dim)
            if student_dim != teacher_dim else nn.Identity()
        )

    def forward(self, x):
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 单 epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(stage, student, teacher, regressor,
              loader, optimizer, device, epoch, is_train):

    student.train() if is_train else student.eval()
    tag = 'Train' if is_train else 'Val  '

    total_loss = correct = total = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(loader, desc=f"[{tag}] Epoch {epoch:03d}", leave=False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for eeg, pcc, audio, vision, labels in pbar:
            eeg, pcc, audio, vision, labels = (
                eeg.to(device), pcc.to(device),
                audio.to(device), vision.to(device), labels.to(device))

            with torch.no_grad():
                t_out = teacher(eeg, pcc, audio, vision)

            s_out = student(eeg, pcc)

            if stage == 1:
                # FitNets hint loss：对齐 Teacher 的多模态 fused 表示
                loss = F.mse_loss(
                    regressor(s_out['eeg_feat']),
                    t_out['fused'].detach(),
                )
            else:
                # Stage 2：CE + KD 软标签
                l_ce = F.cross_entropy(s_out['logits'], labels)
                l_kd = F.kl_div(
                    F.log_softmax(s_out['logits'] / TEMPERATURE, dim=-1),
                    F.softmax(t_out['logits'].detach() / TEMPERATURE, dim=-1),
                    reduction='batchmean',
                ) * (TEMPERATURE ** 2)
                loss = l_ce + l_kd

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            B           = labels.size(0)
            total      += B
            total_loss += loss.item() * B
            if stage == 2:
                preds = s_out['logits'].argmax(1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            post = {'loss': f"{total_loss / total:.4f}"}
            if stage == 2:
                post['acc'] = f"{correct / total:.4f}"
            pbar.set_postfix(post)

    avg_loss = total_loss / total
    avg_acc  = correct / total if stage == 2 else None
    avg_f1   = None
    if stage == 2 and len(all_preds) > 0:
        avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, avg_acc, avg_f1


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    print("\n📦 Loading data ...")
    train_loader, val_loader, test_loader = build_loaders()

    # ── Teacher（加载预训练权重，全程冻结）────────────────────────────────────
    teacher = TeacherModel(
        audio_input_dim  = AUDIO_DIM,
        vision_input_dim = VISION_DIM,
        av_hidden_dim    = AV_HIDDEN,
        av_num_layers    = LSTM_LAYERS,
        av_dropout       = DROPOUT,
        num_nodes        = NUM_NODES,
        eeg_in_features  = IN_FEATURES,
        gcn_hidden       = GCN_HIDDEN,
        gcn_out          = GCN_OUT,
        lstm_hidden      = LSTM_HIDDEN,
        lstm_layers      = LSTM_LAYERS,
        eeg_dropout      = DROPOUT,
        fc_hidden        = FC_HIDDEN,
        num_classes      = NUM_CLASSES,
    ).to(device)

    ckpt = torch.load(TEACHER_CKPT, map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher loaded from {TEACHER_CKPT}  "
          f"(best val acc: {ckpt.get('best_val_acc', '?'):.4f})")

    # ── Student ───────────────────────────────────────────────────────────────
    student = ST_GCLSTM(
        num_nodes   = NUM_NODES,
        in_features = IN_FEATURES,
        gcn_hidden  = GCN_HIDDEN,
        gcn_out     = GCN_OUT,
        lstm_hidden = LSTM_HIDDEN,
        lstm_layers = LSTM_LAYERS,
        fc_hidden   = FC_HIDDEN,
        num_classes = NUM_CLASSES,
        dropout     = DROPOUT,
    ).to(device)
    print(f"Student params: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")

    # regressor：student eeg_feat [B,128] → teacher fused [B,384]
    regressor = HintRegressor(
        student_dim = STUDENT_DIM,   # 128
        teacher_dim = TEACHER_DIM,   # 384
    ).to(device)
    print(f"Regressor: {STUDENT_DIM} → {TEACHER_DIM}")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 1：Hint layer 预训练
    # ════════════════════════════════════════════════════════════════════════
    for p in student.classifier.parameters():
        p.requires_grad_(False)

    opt1 = torch.optim.AdamW(
        list(student.sgcn.parameters())
        + list(student.abilstm.parameters())
        + list(regressor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    print(f"\n{'='*60}")
    print("  Stage 1 — Hint pre-training  "
          f"(eeg_feat → fused MSE,  max {EPOCHS_STAGE1} epochs)")
    print(f"{'='*60}")

    best_hint_loss = float('inf')
    hint_patience  = 0
    hint_ckpt      = os.path.join(SAVE_DIR, 'best_hint_regressor.pth')

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        tr_loss, _, _ = run_epoch(1, student, teacher, regressor,
                                  train_loader, opt1, device, epoch, is_train=True)
        va_loss, _, _ = run_epoch(1, student, teacher, regressor,
                                  val_loader,   opt1, device, epoch, is_train=False)

        flag = ''
        if va_loss < best_hint_loss:
            best_hint_loss = va_loss
            hint_patience  = 0
            torch.save({
                'epoch'          : epoch,
                'student_encoder': {
                    'sgcn'   : student.sgcn.state_dict(),
                    'abilstm': student.abilstm.state_dict(),
                },
                'regressor'      : regressor.state_dict(),
                'best_hint_loss' : best_hint_loss,
            }, hint_ckpt)
            flag = '  ✅'
        else:
            hint_patience += 1
            flag = f'  (patience {hint_patience}/{PATIENCE})'

        print(f"  Epoch {epoch:3d}/{EPOCHS_STAGE1} "
              f"| tr_hint={tr_loss:.4f}  va_hint={va_loss:.4f}{flag}")

        if hint_patience >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

    # 加载 Stage 1 最优编码器权重，作为 Stage 2 初始化
    best_s1 = torch.load(hint_ckpt, map_location=device)
    student.sgcn.load_state_dict(best_s1['student_encoder']['sgcn'])
    student.abilstm.load_state_dict(best_s1['student_encoder']['abilstm'])
    print(f"\nStage 1 best va_hint={best_hint_loss:.4f}  →  loaded for Stage 2")

    # ════════════════════════════════════════════════════════════════════════
    # Stage 2：CE + KD 软标签，端到端微调
    # ════════════════════════════════════════════════════════════════════════
    for p in student.parameters():
        p.requires_grad_(True)

    opt2 = torch.optim.AdamW(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=EPOCHS_STAGE2, eta_min=LR * 0.01,
    )

    best_val_acc   = 0.0
    patience_count = 0
    ckpt_path      = os.path.join(SAVE_DIR, 'best_student_fitnets.pth')

    print(f"\n{'='*60}")
    print("  Stage 2 — Full KD fine-tuning  "
          f"(CE + soft-label KD,  {EPOCHS_STAGE2} epochs)")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5}  {'LR':>8}  "
          f"{'Tr-Loss':>8} {'Tr-Acc':>7} {'Tr-F1':>7}  "
          f"{'Va-Loss':>8} {'Va-Acc':>7} {'Va-F1':>7}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(2, student, teacher, regressor,
                                           train_loader, opt2, device, epoch, is_train=True)
        va_loss, va_acc, va_f1 = run_epoch(2, student, teacher, regressor,
                                           val_loader,   opt2, device, epoch, is_train=False)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        flag = ''
        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            torch.save({'epoch'       : epoch,
                        'model_state' : student.state_dict(),
                        'best_val_acc': best_val_acc}, ckpt_path)
            flag = '  ✅'
        else:
            patience_count += 1
            flag = f'  (patience {patience_count}/{PATIENCE})'

        print(f"  {epoch:5d}  {lr_now:8.2e}  "
              f"{tr_loss:8.4f} {tr_acc:7.4f} {tr_f1:7.4f}  "
              f"{va_loss:8.4f} {va_acc:7.4f} {va_f1:7.4f}{flag}")

        if patience_count >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print(f"\n🔍 Best val acc: {best_val_acc:.4f}  →  {ckpt_path}")
    student.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])

    te_loss, te_acc, te_f1 = run_epoch(2, student, teacher, regressor,
                                        test_loader, None, device, 0, is_train=False)
    print(f"\n{'='*40}")
    print(f"  Test Acc  : {te_acc:.4f}")
    print(f"  Test F1   : {te_f1:.4f}")
    print(f"  Test Loss : {te_loss:.4f}")
    print(f"{'='*40}\n")


if __name__ == '__main__':
    main()
