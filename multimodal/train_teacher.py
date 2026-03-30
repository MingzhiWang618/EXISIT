import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict
from dataset.dataset import CrossSubjectMultiModalDataset
from multimodal.model.Teacher import TeacherModel, TeacherLoss


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
EPOCHS       = 150
LR           = 5e-5
WEIGHT_DECAY = 1e-4

# 模型
NUM_NODES    = 30
IN_FEATURES  = 5
GCN_HIDDEN   = 64
GCN_OUT      = 64
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 1
FC_HIDDEN    = 64
DROPOUT      = 0.5
AV_HIDDEN    = 64
DK           = 32

# 训练
W_AV         = 0.3
W_EEG        = 0.7
PATIENCE     = 20
SAVE_DIR     = './checkpoints'
DEVICE       = 'cuda'


# ═══════════════════════════════════════════════════════════════════════════════
# 多模态 Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalDataset(Dataset):
    def __init__(self,
                 eeg:            np.ndarray,
                 pcc:            np.ndarray,
                 audio:          np.ndarray,
                 vision:         list,
                 labels:         np.ndarray,
                 vision_max_len: Optional[int] = None):

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


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def build_loaders(vision_max_len: Optional[int] = None):
    manager = CrossSubjectMultiModalDataset(
        DATA_ROOT,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )

    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True,
                               normalize=True, compute_pcc=True)

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
# 单 epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, is_train, epoch):
    model.train() if is_train else model.eval()
    tag = 'Train' if is_train else 'Val  '

    total_loss = av_correct = eeg_correct = total = 0
    pbar = tqdm(loader, desc=f"[{tag}] Epoch {epoch:03d}", leave=False)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for eeg, pcc, audio, vision, labels in pbar:
            eeg, pcc, audio, vision, labels = (
                eeg.to(device), pcc.to(device),
                audio.to(device), vision.to(device), labels.to(device))

            out  = model(eeg, pcc, audio, vision)
            loss = criterion(out, labels)['loss']

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            B            = labels.size(0)
            total       += B
            total_loss  += loss.item() * B
            av_correct  += (out['av_logits'].argmax(1)  == labels).sum().item()
            eeg_correct += (out['eeg_logits'].argmax(1) == labels).sum().item()

            pbar.set_postfix({
                'loss'   : f"{total_loss / total:.4f}",
                'av'     : f"{av_correct  / total:.4f}",
                'eeg'    : f"{eeg_correct / total:.4f}",
            })

    return total_loss / total, av_correct / total, eeg_correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    print("\n📦 Loading data ...")
    train_loader, val_loader, test_loader = build_loaders()

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = TeacherModel(
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
        dk               = DK,
        fc_hidden        = FC_HIDDEN,
        num_classes      = NUM_CLASSES,
    ).to(device)

    print(f"Params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = TeacherLoss(w_av=W_AV, w_eeg=W_EEG)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_eeg_acc   = 0.0
    patience_count = 0
    ckpt_path      = os.path.join(SAVE_DIR, 'best_teacher.pth')

    print(f"\n{'='*65}")
    print(f"  {'Epoch':>5}  {'LR':>8}  "
          f"{'Tr-Loss':>8} {'Tr-AV':>7} {'Tr-EEG':>7}  "
          f"{'Va-Loss':>8} {'Va-AV':>7} {'Va-EEG':>7}")
    print(f"{'='*65}")

    for epoch in range(1, EPOCHS + 1):

        tr_loss, tr_av, tr_eeg = run_epoch(
            model, train_loader, criterion, optimizer,
            device, is_train=True,  epoch=epoch)

        va_loss, va_av, va_eeg = run_epoch(
            model, val_loader,   criterion, optimizer,
            device, is_train=False, epoch=epoch)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        flag = ''
        if va_eeg > best_eeg_acc:
            best_eeg_acc   = va_eeg
            patience_count = 0
            torch.save({'epoch'       : epoch,
                        'model_state' : model.state_dict(),
                        'optimizer'   : optimizer.state_dict(),
                        'best_eeg_acc': best_eeg_acc,
                        'val_av_acc'  : va_av}, ckpt_path)
            flag = '  ✅'
        else:
            patience_count += 1
            flag = f'  (patience {patience_count}/{PATIENCE})'

        print(f"  {epoch:5d}  {lr_now:8.2e}  "
              f"{tr_loss:8.4f} {tr_av:7.4f} {tr_eeg:7.4f}"
              f"{va_loss:8.4f} {va_av:7.4f} {va_eeg:7.4f}"
              f"{flag}")

        if patience_count >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print(f"\n🔍 Best val EEG acc: {best_eeg_acc:.4f}  →  {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])

    te_loss, te_av, te_eeg = run_epoch(
        model, test_loader, criterion, None,
        device, is_train=False, epoch=0)

    print(f"\n{'='*40}")
    print(f"  Test AV  Acc : {te_av:.4f}")
    print(f"  Test EEG Acc : {te_eeg:.4f}")
    print(f"  Test Loss    : {te_loss:.4f}")
    print(f"{'='*40}\n")


if __name__ == '__main__':
    main()