import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional, Dict, List, Tuple

# 加入路径以便加载 dataset
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# ── 项目内模块 ─────────────────────────────────────────────────────────────────
from KDbaseline.model.Student import ST_GCLSTM
from dataset.within_subject_dataset import create_within_subject_dataloaders


# =============================================================================
# 结果记录器
# =============================================================================

class ResultRecorder:
    def __init__(self, subjects: List[int]):
        self.subjects = subjects
        self.results  = {sid: {} for sid in subjects}

    def add_result(self, subject_id: int, y_true: np.ndarray, y_pred: np.ndarray):
        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.results[subject_id] = {'y_true': y_true, 'y_pred': y_pred, 'acc': acc, 'f1': f1}

    def get_subject_result(self, subject_id: int) -> Dict:
        return self.results.get(subject_id, {})

    def get_average_metrics(self) -> Dict[str, float]:
        accs = [r['acc'] for r in self.results.values() if 'acc' in r]
        f1s  = [r['f1'] for r in self.results.values() if 'f1' in r]
        return {
            'avg_acc': np.mean(accs) if accs else 0.0,
            'avg_f1': np.mean(f1s) if f1s else 0.0,
            'std_acc': np.std(accs) if len(accs) > 1 else 0.0,
            'std_f1': np.std(f1s) if len(f1s) > 1 else 0.0,
        }

    def print_summary(self):
        print(f"\n{'='*60}")
        print("  Within-Subject Student Training Results")
        print(f"{'='*60}")
        print(f"  {'Subject':>6}  {'Accuracy':>10}  {'F1-Score':>10}")
        print(f"{'='*60}")
        
        for sid in self.subjects:
            if sid in self.results and 'acc' in self.results[sid]:
                r = self.results[sid]
                print(f"  {sid:6d}  {r['acc']:10.4f}  {r['f1']:10.4f}")
        
        avg = self.get_average_metrics()
        print(f"{'='*60}")
        print(f"  {'Average':>6}  {avg['avg_acc']:10.4f}  {avg['avg_f1']:10.4f}")
        print(f"  {'Std':>6}  {avg['std_acc']:10.4f}  {avg['std_f1']:10.4f}")
        print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

DATA_ROOT = {
    'eeg': '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
}

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

EPOCHS       = 150
PATIENCE     = 20
SAVE_DIR     = './checkpoints'
DEVICE       = 'cuda'


# ═══════════════════════════════════════════════════════════════════════════════
# 单 epoch 训练/验证
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer=None, device='cuda'):
    """
    运行单个 epoch 的训练或验证
    
    参数：
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器（训练时提供）
        device: 设备
    
    返回：
        loss, acc, f1
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Train' if is_train else 'Val', leave=False)
    
    with torch.set_grad_enabled(is_train):
        for batch in pbar:
            if len(batch) == 3:  # 包含 PCC
                x, pcc, y = batch
                x, pcc, y = x.to(device), pcc.to(device), y.to(device)
                outputs = model(x, pcc)
            else:  # 不包含 PCC
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x)
            
            loss = criterion(outputs['logits'], y)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 计算指标
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            
            preds = outputs['logits'].argmax(1)
            correct += (preds == y).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # 更新进度条
            acc = correct / total
            pbar.set_postfix({'loss': f"{total_loss/total:.4f}", 'acc': f"{acc:.4f}"})
    
    avg_loss = total_loss / total
    avg_acc = correct / total
    avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, avg_acc, avg_f1


# ═══════════════════════════════════════════════════════════════════════════════
# 训练单个被试
# ═══════════════════════════════════════════════════════════════════════════════

def train_subject(
    subject_id: int,
    data_root_dict: Dict[str, str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    训练单个被试的 Student 模型
    
    参数：
        subject_id: 被试 ID
        data_root_dict: 数据根目录字典
        device: 设备
    
    返回：
        (y_true, y_pred, best_epoch, elapsed_seconds)
    """
    t0 = time.time()

    # ── 数据加载 ─────────────────────────────────────────────────────────────
    print(f"  Loading data...")
    train_loader, test_loader = create_within_subject_dataloaders(
        data_root_dict      = data_root_dict,
        subject_id          = subject_id,
        modality            = 'eeg',
        batch_size          = BATCH_SIZE,
        extract_de          = True,
        segment_1s          = True,
        normalize           = True,
        compute_pcc         = True,
        fs                  = 100,
    )

    # ── 模型初始化 ──────────────────────────────────────────────────────────
    print(f"  Initializing Student model...")
    model = ST_GCLSTM(
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
    
    # ── 优化器和损失函数 ────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01,
    )
    criterion = nn.CrossEntropyLoss()

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    patience_count = 0
    ckpt_path = os.path.join(SAVE_DIR, f'best_student_s{subject_id:02d}.pth')
    
    print(f"  Training Student for subject {subject_id}...")
    
    for epoch in range(1, EPOCHS + 1):
        # 训练
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 验证
        va_loss, va_acc, va_f1 = run_epoch(
            model, test_loader, criterion, None, device
        )
        
        # 学习率调度
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        
        # 早停和保存最佳模型
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            patience_count = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'best_val_acc': best_val_acc,
            }, ckpt_path)
        else:
            patience_count += 1
        
        # 打印进度
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.4f}, Val Acc={va_acc:.4f}")
        
        # 早停
        if patience_count >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    model.eval()
    
    # 收集测试结果
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                x, pcc, y = batch
                x, pcc = x.to(device), pcc.to(device)
                outputs = model(x, pcc)
            else:
                x, y = batch
                x = x.to(device)
                outputs = model(x)
            
            preds = outputs['logits'].argmax(1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    elapsed = time.time() - t0
    
    return y_true, y_pred, best_epoch, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数：运行所有被试的实验
# ═══════════════════════════════════════════════════════════════════════════════

def run_within_subject_experiment(
    data_root_dict: Dict[str, str],
    subjects: List[int] = None,
    device: Optional[torch.device] = None,
):
    """
    运行 within-subject 实验
    
    参数：
        data_root_dict: 数据根目录字典
        subjects: 被试列表，默认 1-42
        device: 设备，默认自动选择
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if subjects is None:
        subjects = list(range(1, 43))
    
    recorder = ResultRecorder(subjects)
    
    for subject_id in subjects:
        print(f"\n{'='*60}")
        print(f"  Subject {subject_id:02d} / {len(subjects)}")
        print(f"{'='*60}")
        
        try:
            y_true, y_pred, best_epoch, elapsed = train_subject(
                subject_id=subject_id,
                data_root_dict=data_root_dict,
                device=device,
            )
            
            # 记录结果
            recorder.add_result(subject_id, y_true, y_pred)
            
            # 打印被试结果
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            print(f"  🎯 Test Acc: {acc:.4f}, F1: {f1:.4f}")
            print(f"  🏆 Best Epoch: {best_epoch}")
            print(f"  ⏱️  Elapsed: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    recorder.print_summary()


if __name__ == '__main__':
    run_within_subject_experiment(
        data_root_dict=DATA_ROOT,
        subjects=list(range(1, 43)),
    )
