"""
ST-GCLSTM Hierarchical Training Script (With Region Importance)
================================================================
1. 适配分层脑区架构（Intra-Region -> Inter-Region -> BiLSTM）
2. 支持 Region Importance Gate 和 Temporal Attention
3. 自动根据 MY_REGION_DEFINITION 生成节点索引
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

# 添加项目根目录到Python路径
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# 本地模块
from model.ST_GCLSTM_region import ST_GCLSTM  # 新模型
from dataset.dataset import create_single_modality_dataloaders


# =============================================================================
# 配置
# =============================================================================
class Config:
    # 数据
    data_root_dict = {
        'audio':  '/data2/mingzhi/BCI/dataset/EAV_old/Audio',
        'eeg':    '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision1',
    }
    modality    = 'eeg'
    extract_de  = True
    segment_1s  = True
    normalize   = True
    fs          = 100

    # 模型参数
    in_features = 5          # DE 频段数
    gcn_dim     = 64
    lstm_hidden = 64
    num_classes = 5
    dropout     = 0.5

    # 训练参数
    batch_size   = 64
    epochs       = 100
    lr           = 1e-4
    weight_decay = 1e-3
    patience     = 30

    # 输出
    save_dir = './checkpoints'
    exp_name = 'st_gclstm_hierarchical_regiongate'

    # 设备
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
    torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count


# =============================================================================
# 训练与评估函数 (兼容 region_weights)
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        
        logits, _, _, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return {'loss': loss_meter.avg,
            'acc': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)}

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _, _ = model(x)
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
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

    # --- 数据加载 ---
    print("\nLoading EEG Data (Hierarchical Ready)...")
    train_loader, val_loader, test_loader = create_single_modality_dataloaders(
        data_root_dict = cfg.data_root_dict,
        modality       = cfg.modality,
        batch_size     = cfg.batch_size,
        extract_de     = cfg.extract_de,
        segment_1s     = cfg.segment_1s,
        normalize      = cfg.normalize,
        fs             = cfg.fs,
    )

    # --- 模型初始化 ---
    model = ST_GCLSTM(
        in_features = cfg.in_features,
        gcn_dim     = cfg.gcn_dim,
        lstm_hidden = cfg.lstm_hidden,
        num_classes = cfg.num_classes,
        dropout     = cfg.dropout
    ).to(cfg.device)

    print(f"Model    : ST-GCLSTM (Hierarchical + RegionGate)")
    print(f"Device   : {cfg.device}")
    print(f"Params   : {count_parameters(model):,}")
    print(f"Regions  : {model.num_regions} predefined blocks")

    # --- 损失 & 优化器 ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    # --- 训练循环 ---
    best_val_acc = 0.0
    patience_count = 0

    print("\nEpoch | Train Loss  Train Acc | Val Loss   Val Acc | LR      Time(s)")
    print("-"*80)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_m   = evaluate(model, val_loader, criterion, cfg.device)

        scheduler.step(val_m['acc'])
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>5} | {train_m['loss']:>10.4f} {train_m['acc']:>10.4f} | "
              f"{val_m['loss']:>10.4f} {val_m['acc']:>10.4f} | {current_lr:.2e} {elapsed:>5.1f}s")

        # --- 保存最优模型 ---
        if val_m['acc'] > best_val_acc:
            best_val_acc = val_m['acc']
            patience_count = 0
            torch.save({'model_state': model.state_dict(), 'cfg': cfg.__dict__}, ckpt_path)
            print(f"    --> Saved Best Model!")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # --- 测试 ---
    print("\nLoading best model for testing...")
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt['model_state'])
    test_m = evaluate(model, test_loader, criterion, cfg.device)
    print(f"TEST RESULTS: Acc={test_m['acc']:.4f}, F1={test_m['f1']:.4f}")
    print("Confusion Matrix:\n", test_m['cm'])


if __name__ == '__main__':
    train(Config())