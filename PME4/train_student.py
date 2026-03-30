"""
ST-GCLSTM Training Script — PME4 Dataset (Student Model Version)
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
from PME4.model.Student import ST_GCLSTM
from PME4.dataset.dataset import create_pme4_dataloaders

# =============================================================================
# 配置
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root   = '/data2/zhiwen/dataset/PME4'
    extract_de  = True
    normalize   = True
    compute_pcc = True        # Student 模型需要输入 pcc
    fs          = 1000        # PME4 采样率

    # ── 模型超参数 (已根据 Student 类默认值对齐) ──────────────────────────────────
    num_nodes   = 32          # 实际运行时会自动从数据中修正
    in_features = 5           # DE 特征通常为 5 个频带
    gcn_hidden  = 64
    gcn_out     = 64
    lstm_hidden = 128
    lstm_layers = 2
    fc_hidden   = 256
    num_classes = 4           
    dropout     = 0.5

    # ── 训练参数 ──────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 1e-4
    weight_decay = 1e-3
    patience     = 30

    # ── 设备与存储 ────────────────────────────────────────────────────────────
    save_dir = './checkpoints'
    exp_name = 'st_gclstm_student_pme4'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed   = 2024


# =============================================================================
# 工具类与函数
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# =============================================================================
# 核心训练逻辑
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, pcc, y in loader:
        x, pcc, y = x.to(device), pcc.to(device), y.to(device)

        optimizer.zero_grad()
        
        # --- 关键修改：处理 Student 模型的字典输出 ---
        outputs = model(x, pcc)
        logits = outputs['logits']  # 提取分类 logits
        
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


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    for x, pcc, y in loader:
        x, pcc, y = x.to(device), pcc.to(device), y.to(device)

        # --- 关键修改：提取 logits ---
        outputs = model(x, pcc)
        logits = outputs['logits']
        
        loss = criterion(logits, y)

        loss_meter.update(loss.item(), x.size(0))
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm  = confusion_matrix(all_labels, all_preds)
    return {'loss': loss_meter.avg, 'acc': acc, 'f1': f1, 'cm': cm}


def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f'{cfg.exp_name}_best.pt')

    # 1. 加载数据
    print(f"\n{'='*20} Loading Data {'='*20}")
    # 注意：此处假设 create_pme4_dataloaders 已经导入
    train_loader, val_loader, test_loader = create_pme4_dataloaders(
        root        = cfg.data_root,
        batch_size  = cfg.batch_size,
        extract_de  = cfg.extract_de,
        normalize   = cfg.normalize,
        compute_pcc = cfg.compute_pcc,
        fs          = cfg.fs,
        num_workers = 4,
    )

    # 自动校准节点数
    sample_x, _, _ = next(iter(train_loader))
    if sample_x.shape[2] != cfg.num_nodes:
        print(f"Correction: Changing num_nodes from {cfg.num_nodes} to {sample_x.shape[2]}")
        cfg.num_nodes = sample_x.shape[2]

    # 2. 初始化 Student 模型
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

    # 3. 损失函数与优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,2], dtype=torch.float).to(cfg.device))
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    # 4. 训练循环
    print(f"\n{'='*20} Starting Training {'='*20}")
    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_res   = evaluate(model, val_loader, criterion, cfg.device)
        
        scheduler.step(val_res['acc'])
        elapsed = time.time() - t0

        print(f"Epoch {epoch:>3} | Train Loss: {train_res['loss']:.4f} Acc: {train_res['acc']:.4f} | "
              f"Val Loss: {val_res['loss']:.4f} Acc: {val_res['acc']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if val_res['acc'] > best_val_acc:
            best_val_acc = val_res['acc']
            patience_count = 0
            torch.save({
                'model_state': model.state_dict(),
                'cfg': vars(cfg),
                'val_acc': best_val_acc
            }, ckpt_path)
            print("  >>> Best model saved.")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 5. 测试
    print(f"\n{'='*20} Final Testing {'='*20}")
    ckpt = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    test_res = evaluate(model, test_loader, criterion, cfg.device)
    label_names = ['neg', 'pos']
    print(f"Test Accuracy: {test_res['acc']:.4f}")
    print(f"Test F1 Score: {test_res['f1']:.4f}")
    print(f"\n  Confusion Matrix ({' / '.join(label_names)}):")
    print(test_res['cm'])

if __name__ == '__main__':
    cfg = Config()
    train(cfg)