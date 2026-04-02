import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Optional, Dict, List, Tuple

# 加入路径以便加载 dataset
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# ── 项目内模块 ─────────────────────────────────────────────────────────────────
from KDbaseline.model.AMBOKD_Model import AMBOKDModel
from dataset.within_subject_dataset import create_within_subject_dataloaders_multimodal


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
        print("  Within-Subject Results Summary")
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
    lr           = 5e-4
    weight_decay = 1e-3
    patience     = 20
    seed         = 2024
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── AMBOKD 超参 ───────────────────────────────────────────────────────────
    temperature = 4.0   # 软标签温度，常用 2~6

    # ── 路径 ──────────────────────────────────────────────────────────────────
    save_dir    = './checkpoints'
    model_ckpt  = './checkpoints/best_ambokd_model.pth'


# =============================================================================
# 工具
# =============================================================================

def set_seed(seed):
    import numpy as np
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

def run_phase(cfg, model, loader, optimizer=None, epoch=1):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    # 增加对 EEG 和 AV 分支的 Loss 监控
    meters = {k: AverageMeter() for k in ('loss', 'ce_fused', 'ce_eeg', 'ce_av')}
    
    # 分别记录 Fused 分支和 EEG 分支的预测结果
    preds_fused, preds_eeg, labels_list = [], [], []

    with torch.set_grad_enabled(is_train):
        for eeg, pcc, audio, vision, y in loader:
            eeg, pcc, audio, vision, y = [d.to(cfg.device) for d in (eeg, pcc, audio, vision, y)]
            
            # 模型内置了 AMBOKD Loss 计算，返回字典包含各分支 logits
            out = model(eeg, pcc, audio, vision, labels=y, epoch=epoch)
            
            if is_train:
                optimizer.zero_grad()
                out['loss'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # 更新各项指标
            batch_size = y.size(0)
            meters['loss'].update(out['loss'].item(), batch_size)
            meters['ce_fused'].update(out['ce_fused'].item(), batch_size)
            meters['ce_eeg'].update(out['ce_eeg'].item(), batch_size)
            if 'ce_av' in out:
                meters['ce_av'].update(out['ce_av'].item(), batch_size)
            
            # 收集 Fused 分支和 EEG 单模态分支的预测
            preds_fused.extend(out['logits_fused'].argmax(1).cpu().numpy())
            preds_eeg.extend(out['logits_eeg'].argmax(1).cpu().numpy())
            labels_list.extend(y.cpu().numpy())
            
    # 计算融合后的准确率
    acc_fused = accuracy_score(labels_list, preds_fused)
    f1_fused  = f1_score(labels_list, preds_fused, average='weighted', zero_division=0)
    
    # 计算 EEG 单模态的准确率
    acc_eeg   = accuracy_score(labels_list, preds_eeg)
    f1_eeg    = f1_score(labels_list, preds_eeg, average='weighted', zero_division=0)

    return {
        'acc': acc_fused,
        'f1': f1_fused,
        'acc_eeg': acc_eeg,
        'f1_eeg': f1_eeg,
        'loss': meters['loss'].avg,
        'ce_fused': meters['ce_fused'].avg,
        'ce_eeg': meters['ce_eeg'].avg
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 训练单个被试
# ═══════════════════════════════════════════════════════════════════════════════

def train_subject(
    subject_id: int,
    data_root_dict: Dict[str, str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """
    训练单个被试的 AMBOKD 模型
    
    参数：
        subject_id: 被试 ID
        data_root_dict: 数据根目录字典
        device: 设备
    
    返回：
        (y_true, teacher_pred, student_pred, best_epoch_teacher, best_epoch_student, elapsed_seconds)
    """
    t0 = time.time()

    # ── 数据加载 ─────────────────────────────────────────────────────────────
    print(f"  Loading data...")
    train_loader, test_loader = create_within_subject_dataloaders_multimodal(
        data_root_dict      = data_root_dict,
        subject_id          = subject_id,
        batch_size          = 64,
        extract_de          = True,
        segment_1s          = True,
        normalize           = True,
        compute_pcc         = True,
        fs                  = 100,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    print("  Initializing AMBOKD Model...")
    cfg = Config()
    model = AMBOKDModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    
    best_val_acc_eeg = 0.0
    patience_count = 0
    ckpt_path = os.path.join(cfg.save_dir, f'best_ambokd_s{subject_id:02d}.pth')

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f"  AMBOKD Training  seed={cfg.seed}")

    for epoch in range(1, cfg.epochs + 1):
        t0_epoch = time.time()

        tr = run_phase(cfg, model, train_loader, optimizer, epoch)
        va = run_phase(cfg, model, test_loader, epoch=epoch)

        scheduler.step(va['acc_eeg'])
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0_epoch

        if va['acc_eeg'] > best_val_acc_eeg:
            best_val_acc_eeg = va['acc_eeg']
            patience_count = 0
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({'epoch'      : epoch,
                        'model_state': model.state_dict(),
                        'val_acc_eeg': best_val_acc_eeg},
                       ckpt_path)
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)['model_state'])
    model.eval()

    # 收集测试结果
    y_true = []
    fused_pred = []
    eeg_pred = []
    
    with torch.no_grad():
        for eeg, pcc, audio, vision, y in test_loader:
            eeg, pcc, audio, vision, y = [d.to(device) for d in (eeg, pcc, audio, vision, y)]
            out = model(eeg, pcc, audio, vision, labels=y, epoch=0)
            
            fused_pred.extend(out['logits_fused'].argmax(1).cpu().numpy())
            eeg_pred.extend(out['logits_eeg'].argmax(1).cpu().numpy())
            y_true.extend(y.cpu().numpy())

    y_true = np.array(y_true)
    fused_pred = np.array(fused_pred)
    eeg_pred = np.array(eeg_pred)
    elapsed = time.time() - t0

    # 对于 AMBOKD，我们使用 EEG 分支的结果作为 student_pred
    return y_true, fused_pred, eeg_pred, 0, 0, elapsed


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
    import numpy as np
    
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
            y_true, fused_pred, eeg_pred, best_epoch_teacher, best_epoch_student, elapsed = train_subject(
                subject_id=subject_id,
                data_root_dict=data_root_dict,
                device=device,
            )
            
            # 记录结果（使用 EEG 分支的结果）
            recorder.add_result(subject_id, y_true, eeg_pred)
            
            # 打印被试结果
            fused_acc = accuracy_score(y_true, fused_pred)
            eeg_acc = accuracy_score(y_true, eeg_pred)
            fused_f1 = f1_score(y_true, fused_pred, average='weighted', zero_division=0)
            eeg_f1 = f1_score(y_true, eeg_pred, average='weighted', zero_division=0)
            
            print(f"  🎯 Fused Acc: {fused_acc:.4f}, F1: {fused_f1:.4f}")
            print(f"  🎯 EEG Acc: {eeg_acc:.4f}, F1: {eeg_f1:.4f}")
            print(f"  ⏱️  Elapsed: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印汇总结果
    recorder.print_summary()


if __name__ == "__main__":
    cfg = Config()
    run_within_subject_experiment(
        data_root_dict=cfg.data_root_dict,
        subjects=list(range(1, 43)),
    )
