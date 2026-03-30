"""
ST-GCLSTM Training Script
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from multimodal.model.Student import ST_GCLSTM
from dataset.dataset import create_single_modality_dataloaders


# =============================================================================
# 配置
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root_dict = {
        'audio':  '/data2/mingzhi/BCI/dataset/EAV_old/Audio',
        'eeg':    '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision1',
    }
    modality    = 'eeg'
    extract_de  = True
    segment_1s  = True
    normalize   = True
    compute_pcc = True       # DataLoader 每批返回 (x, pcc, y)
    fs          = 100

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

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 1e-3
    weight_decay = 1e-3
    patience     = 30

    # ── 输出 ──────────────────────────────────────────────────────────────────
    save_dir  = './checkpoints'
    exp_name  = 'st_gclstm_eav'

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 2024


# =============================================================================
# 工具函数
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# =============================================================================
# 单 epoch 训练
# =============================================================================

def train_one_epoch(model:     nn.Module,
                    loader:    DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device:    str) -> dict:
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []
    for x, pcc, y in loader:
        print("x shape:", x.shape)
        print("x mean/std:", x.mean().item(), x.std().item())
        print("y:", y[:10])
        break
    for x, pcc, y in loader:          # ← 解包三个值
        x   = x.to(device)            # [B, T, N, F]
        pcc = pcc.to(device)          # [B, T, N, N]
        y   = y.to(device)            # [B]

        optimizer.zero_grad()
        output = model(x, pcc)     # ← 把 pcc 传给模型
        logits = output['logits']
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {'loss': loss_meter.avg, 'acc': acc, 'f1': f1}


# =============================================================================
# 评估
# =============================================================================

@torch.no_grad()
def evaluate(model:     nn.Module,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    str) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, pcc, y in loader:          # ← 解包三个值
        x   = x.to(device)
        pcc = pcc.to(device)          # [B, T, N, N]
        y   = y.to(device)

        output = model(x, pcc)     # ← 把 pcc 传给模型
        logits = output['logits']
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)
    return {'loss': loss_meter.avg, 'acc': acc, 'f1': f1, 'cm': cm}


# =============================================================================
# 主训练流程
# =============================================================================

def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f'{cfg.exp_name}_best.pt')

    # ── 1. 数据加载 ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading data ...")

    # DataLoader 每批返回 (x[B,T,N,F], pcc[B,T,N,N], y[B])
    train_loader, val_loader, test_loader = create_single_modality_dataloaders(
        data_root_dict = cfg.data_root_dict,
        modality       = cfg.modality,
        batch_size     = cfg.batch_size,
        extract_de     = cfg.extract_de,
        segment_1s     = cfg.segment_1s,
        normalize      = cfg.normalize,
        compute_pcc    = cfg.compute_pcc,
        fs             = cfg.fs,
    )

    for x, pcc, y in train_loader:
        print(f"EEG  batch shape : {x.shape}")    # [B, T, N, F]
        print(f"PCC  batch shape : {pcc.shape}")  # [B, T, N, N]
        print(f"Label batch shape: {y.shape}")    # [B]
        break

    # ── 2. 模型初始化 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)

    # 不再需要 init_pcc，PCC 每批动态传入
    model = ST_GCLSTM(
        num_nodes   = cfg.num_nodes,
        in_features = cfg.in_features,
        gcn_hidden  = cfg.gcn_hidden,
        gcn_out     = cfg.gcn_out,
        lstm_hidden = cfg.lstm_hidden,
        lstm_layers = cfg.lstm_layers,
        fc_hidden   = cfg.fc_hidden,
        num_classes = cfg.num_classes,
        dropout     = cfg.dropout,
    ).to(cfg.device)

    print(f"Model  : ST-GCLSTM")
    print(f"Device : {cfg.device}")
    print(f"Params : {count_parameters(model):,}")

    # ── 3. 损失函数 & 优化器 & 调度器 ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5,
        patience=7, min_lr=1e-6)

    # ── 4. 训练循环 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} {'Train F1':>9} "
          f"| {'Val Loss':>9} {'Val Acc':>9} {'Val F1':>8} | {'LR':>8} | {'Time':>6}")
    print("-"*90)

    best_val_acc   = 0.0
    patience_count = 0
    history        = {'train': [], 'val': []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, cfg.device)
        val_metrics   = evaluate(model, val_loader, criterion, cfg.device)

        scheduler.step(val_metrics['acc'])
        elapsed = time.time() - t0

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6} | "
              f"{train_metrics['loss']:>10.4f} "
              f"{train_metrics['acc']:>10.4f} "
              f"{train_metrics['f1']:>9.4f} | "
              f"{val_metrics['loss']:>9.4f} "
              f"{val_metrics['acc']:>9.4f} "
              f"{val_metrics['f1']:>8.4f} | "
              f"{current_lr:>8.2e} | "
              f"{elapsed:>5.1f}s")

        if val_metrics['acc'] > best_val_acc:
            best_val_acc   = val_metrics['acc']
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc':     best_val_acc,
                'val_f1':      val_metrics['f1'],
            }, ckpt_path)
            print(f"         ✅ Best model saved  (val_acc={best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"\n⏹  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs)")
                break

    # ── 5. 测试 ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Loading best checkpoint for testing ...")
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])

    test_metrics = evaluate(model, test_loader, criterion, cfg.device)

    print(f"\n{'='*60}")
    print(f"  Test Accuracy : {test_metrics['acc']:.4f}")
    print(f"  Test F1 Score : {test_metrics['f1']:.4f}")
    print(f"  Test Loss     : {test_metrics['loss']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(test_metrics['cm'])
    print(f"{'='*60}\n")

    return history, test_metrics


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    cfg = Config()
    history, test_metrics = train(cfg)