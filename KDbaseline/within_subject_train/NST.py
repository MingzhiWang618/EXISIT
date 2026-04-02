import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import f1_score, accuracy_score

# 加入路径以便加载 dataset
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# ── 项目内模块 ─────────────────────────────────────────────────────────────────
from KDbaseline.model.Teacher import TeacherModel, TeacherLoss
from KDbaseline.model.Student import ST_GCLSTM
from dataset.within_subject_dataset import create_within_subject_dataloaders, create_within_subject_dataloaders_multimodal


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

EPOCHS       = 150
PATIENCE     = 20
MMD_LAMBDA   = 1.0      # MMD 损失权重
SAVE_DIR     = './checkpoints'
DEVICE       = 'cuda'
TEACHER_CKPT = './checkpoints/best_teacher.pth'

STUDENT_DIM  = LSTM_HIDDEN * 2         # 128
TEACHER_DIM  = LSTM_HIDDEN * 2 + AV_HIDDEN * 4  # 384


# ═══════════════════════════════════════════════════════════════════════════════
# MMD 损失（多项式核）
# ═══════════════════════════════════════════════════════════════════════════════

def mmd_loss(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """
    f_s: [B, D]  Student 特征（经过 regressor 投影后）
    f_t: [B, D]  Teacher fused 特征
    多项式核：k(x, y) = (x·yᵀ / D + 1)²
    """
    D  = f_s.size(1)
    ss = (f_s @ f_s.T / D + 1).pow(2)
    tt = (f_t @ f_t.T / D + 1).pow(2)
    st = (f_s @ f_t.T / D + 1).pow(2)
    return ss.mean() + tt.mean() - 2 * st.mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Regressor：student eeg_feat [B,128] → teacher fused [B,384]
# ═══════════════════════════════════════════════════════════════════════════════

class Regressor(nn.Module):
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = (
            nn.Linear(student_dim, teacher_dim)
            if student_dim != teacher_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 单 epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(student, teacher, regressor,
              loader, optimizer, device, epoch, is_train):

    student.train() if is_train else student.eval()
    tag = 'Train' if is_train else 'Val  '

    total_loss = total_ce = total_mmd = correct = total = 0
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

            l_ce  = F.cross_entropy(s_out['logits'], labels)
            l_mmd = mmd_loss(
                regressor(s_out['eeg_feat']),
                t_out['fused'].detach(),
            )
            loss = l_ce + MMD_LAMBDA * l_mmd

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(regressor.parameters()), 1.0)
                optimizer.step()

            B           = labels.size(0)
            total      += B
            total_loss += loss.item()  * B
            total_ce   += l_ce.item()  * B
            total_mmd  += l_mmd.item() * B
            preds = s_out['logits'].argmax(1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f"{total_loss / total:.4f}",
                'ce'  : f"{total_ce   / total:.4f}",
                'mmd' : f"{total_mmd  / total:.4f}",
                'acc' : f"{correct    / total:.4f}",
            })

    n = total
    avg_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(all_preds) > 0 else 0.0
    return total_loss / n, total_ce / n, total_mmd / n, correct / n, avg_f1


# ═══════════════════════════════════════════════════════════════════════════════
# 训练单个被试
# ═══════════════════════════════════════════════════════════════════════════════

def train_subject(
    subject_id: int,
    data_root_dict: Dict[str, str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """
    训练单个被试的 NST 模型
    
    参数：
        subject_id: 被试 ID
        data_root_dict: 数据根目录字典
        device: 设备
    
    返回：
        (y_true, teacher_pred, student_pred, best_epoch_teacher, best_epoch_student, elapsed_seconds)
    """
    t0 = time.time()

    # ── 数据加载 ─────────────────────────────────────────────────────────────
    # 创建多模态数据加载器
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

    # ── 训练 Teacher ─────────────────────────────────────────────────────────
    print(f"  Training Teacher...")
    
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
    
    best_epoch_teacher = 0
    
    # 加载预训练权重（如果存在）
    if os.path.exists(TEACHER_CKPT):
        ckpt = torch.load(TEACHER_CKPT, map_location=device)
        teacher.load_state_dict(ckpt['model_state'])
        print(f"  Teacher loaded from {TEACHER_CKPT}")
    else:
        # 训练 Teacher
        teacher_optimizer = torch.optim.AdamW(
            teacher.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        teacher_criterion = TeacherLoss()
        
        best_teacher_acc = 0.0
        best_epoch_teacher = 0
        
        print(f"  Training Teacher for subject {subject_id}...")
        for epoch in range(1, 100 + 1):
            teacher.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for eeg, pcc, audio, vision, labels in train_loader:
                eeg, pcc, audio, vision, labels = (
                    eeg.to(device), pcc.to(device),
                    audio.to(device), vision.to(device), labels.to(device)
                )
                
                teacher_optimizer.zero_grad()
                outputs = teacher(eeg, pcc, audio, vision)
                loss_dict = teacher_criterion(outputs, labels)
                loss = loss_dict['loss']
                loss.backward()
                teacher_optimizer.step()
                
                total_loss += loss.item() * labels.size(0)
                preds = outputs['logits'].argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            # 验证
            teacher.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for eeg, pcc, audio, vision, labels in test_loader:
                    eeg, pcc, audio, vision, labels = (
                        eeg.to(device), pcc.to(device),
                        audio.to(device), vision.to(device), labels.to(device)
                    )
                    outputs = teacher(eeg, pcc, audio, vision)
                    preds = outputs['logits'].argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            if val_acc > best_teacher_acc:
                best_teacher_acc = val_acc
                best_epoch_teacher = epoch
                # 保存 Teacher 权重
                os.makedirs(SAVE_DIR, exist_ok=True)
                torch.save({
                    'model_state': teacher.state_dict(),
                    'best_val_acc': best_teacher_acc,
                }, TEACHER_CKPT)
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: Train Loss={total_loss/total:.4f}, Val Acc={val_acc:.4f}")
    
    # 冻结 Teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── 训练 Student (NST) ─────────────────────────────────────────────────
    print(f"  Training Student (NST)...")
    
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

    # ── Regressor ─────────────────────────────────────────────────────────────
    regressor = Regressor(
        student_dim = STUDENT_DIM,   # 128
        teacher_dim = TEACHER_DIM,   # 384
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(regressor.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.01,
    )

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    ckpt_path      = os.path.join(SAVE_DIR, f'best_student_nst_s{subject_id:02d}.pth')

    print(f"  NST Distillation  (CE + MMD,  single stage)")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_ce, tr_mmd, tr_acc, tr_f1 = run_epoch(
            student, teacher, regressor,
            train_loader, optimizer, device, epoch, is_train=True)

        va_loss, va_ce, va_mmd, va_acc, va_f1 = run_epoch(
            student, teacher, regressor,
            test_loader, optimizer, device, epoch, is_train=False)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save({
                'epoch'       : epoch,
                'model_state' : student.state_dict(),
                'regressor'   : regressor.state_dict(),
                'best_val_acc': best_val_acc,
            }, ckpt_path)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    student.load_state_dict(
        torch.load(ckpt_path, map_location=device)['model_state'])
    student.eval()

    # 收集测试结果
    y_true = []
    student_pred = []
    teacher_pred = []
    
    with torch.no_grad():
        for eeg, pcc, audio, vision, labels in test_loader:
            eeg, pcc, audio, vision, labels = (
                eeg.to(device), pcc.to(device),
                audio.to(device), vision.to(device), labels.to(device)
            )
            
            # Teacher 预测
            t_out = teacher(eeg, pcc, audio, vision)
            teacher_pred.extend(t_out['logits'].argmax(1).cpu().numpy())
            
            # Student 预测
            s_out = student(eeg, pcc)
            student_pred.extend(s_out['logits'].argmax(1).cpu().numpy())
            
            y_true.extend(labels.cpu().numpy())

    y_true = np.array(y_true)
    teacher_pred = np.array(teacher_pred)
    student_pred = np.array(student_pred)
    elapsed = time.time() - t0

    return y_true, teacher_pred, student_pred, best_epoch_teacher, 0, elapsed


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
            y_true, teacher_pred, student_pred, best_epoch_teacher, best_epoch_student, elapsed = train_subject(
                subject_id=subject_id,
                data_root_dict=data_root_dict,
                device=device,
            )
            
            # 记录结果
            recorder.add_result(subject_id, y_true, student_pred)
            
            # 打印被试结果
            teacher_acc = accuracy_score(y_true, teacher_pred)
            student_acc = accuracy_score(y_true, student_pred)
            teacher_f1 = f1_score(y_true, teacher_pred, average='weighted', zero_division=0)
            student_f1 = f1_score(y_true, student_pred, average='weighted', zero_division=0)
            
            print(f"  🎯 Teacher Acc: {teacher_acc:.4f}, F1: {teacher_f1:.4f}")
            print(f"  🎯 Student Acc: {student_acc:.4f}, F1: {student_f1:.4f}")
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
