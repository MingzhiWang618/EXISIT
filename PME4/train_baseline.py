"""
PME4 Baseline Training Script
切换模型只需修改 Config.model_name：
    'eegnet'    → EEGNet
    'eegformer' → EEGFormer
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sys

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')
from PME4.dataset.dataset import create_pme4_dataloaders
from PME4.model.EEGNet    import EEGNet
from PME4.model.EEGFormer import EEGFormer


# =============================================================================
# 模型工厂
# =============================================================================

def build_model(cfg, Chans: int, Samples: int) -> nn.Module:
    name = cfg.model_name.lower()
    if name == 'eegnet':
        return EEGNet(
            nb_classes  = cfg.num_classes,
            Chans       = Chans,
            Samples     = Samples,
            dropoutRate = cfg.dropout,
            kernLength  = cfg.kernLength,
            F1          = cfg.F1,
            D           = cfg.D,
            F2          = cfg.F2,
        )
    elif name == 'eegformer':
        return EEGFormer(
            eeg_channel = Chans,
            num_classes = cfg.num_classes,
            dropout     = cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: eegnet | eegformer")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return loss

# =============================================================================
# 配置
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root  = '/data2/zhiwen/dataset/PME4'
    normalize  = True

    # ── 模型选择 ──────────────────────────────────────────────────────────────
    model_name  = 'eegformer'   # ← 改这里: 'eegnet' | 'eegformer'
    num_classes = 2
    dropout     = 0.1

    # EEGNet 专用（model_name='eegnet' 时生效）
    kernLength = 500
    F1         = 8
    D          = 2
    F2         = 16

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 4
    epochs       = 150
    lr           = 1e-3
    weight_decay = 1e-3
    patience     = 30

    # ── 输出（exp_name 自动跟随 model_name） ──────────────────────────────────
    save_dir = './checkpoints'

    @property
    def exp_name(self):
        return f'{self.model_name}_pme4'

    # ── 设备 ──────────────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 2024


# =============================================================================
# 工具
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
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0.0
    def update(self, v, n=1): self.sum += v * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


# =============================================================================
# 训练 / 评估
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        meter.update(loss.item(), x.size(0))
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {'loss': meter.avg, 'acc': acc, 'f1': f1}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits  = model(x)
        loss    = criterion(logits, y)
        meter.update(loss.item(), x.size(0))
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)
    return {'loss': meter.avg, 'acc': acc, 'f1': f1, 'cm': cm}


# =============================================================================
# 主流程
# =============================================================================

def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f'{cfg.exp_name}_best.pt')

    # ── 1. 数据加载 ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Model : {cfg.model_name.upper()}")
    print("Loading PME4 data ...")

    train_loader, val_loader, test_loader = create_pme4_dataloaders(
        root        = cfg.data_root,
        batch_size  = cfg.batch_size,
        segment_1s  = False,
        extract_de  = False,
        normalize   = cfg.normalize,
        compute_pcc = False,
        num_workers = 4,
    )

    x, y = next(iter(train_loader))
    _, Chans, Samples = x.shape
    print(f"\n  EEG batch : {x.shape}   (Chans={Chans}, Samples={Samples})")

    # ── 2. 模型 ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    model = build_model(cfg, Chans, Samples).to(cfg.device)
    print(f"Model  : {cfg.model_name.upper()}")
    print(f"Device : {cfg.device}")
    print(f"Params : {count_parameters(model):,}")

    # ── 3. 损失 & 优化器 & 调度器 ────────────────────────────────────────────
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)

    # ── 4. 训练循环 ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Epoch':>6} | {'Train Loss':>10} {'Train Acc':>10} {'Train F1':>9} "
          f"| {'Val Loss':>9} {'Val Acc':>9} {'Val F1':>8} | {'LR':>8} | {'Time':>6}")
    print("-"*90)

    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        vl = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step(vl['acc'])
        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6} | "
              f"{tr['loss']:>10.4f} {tr['acc']:>10.4f} {tr['f1']:>9.4f} | "
              f"{vl['loss']:>9.4f} {vl['acc']:>9.4f} {vl['f1']:>8.4f} | "
              f"{current_lr:>8.2e} | {elapsed:>5.1f}s")

        if vl['acc'] > best_val_acc:
            best_val_acc   = vl['acc']
            patience_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc':     best_val_acc,
                'val_f1':      vl['f1'],
                'model_name':  cfg.model_name,
                'Chans':       Chans,
                'Samples':     Samples,
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

    te = evaluate(model, test_loader, criterion, cfg.device)

    label_names = ['neg', 'pos']
    print(f"\n  Test Accuracy : {te['acc']:.4f}")
    print(f"  Test F1 Score : {te['f1']:.4f}")
    print(f"  Test Loss     : {te['loss']:.4f}")
    print(f"\n  Confusion Matrix ({' / '.join(label_names)}):")
    print(te['cm'])
    print("="*60)

    return te


# =============================================================================
# 入口
# =============================================================================

if __name__ == '__main__':
    cfg = Config()
    train(cfg)