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
# 训练单个被试
# ═══════════════════════════════════════════════════════════════════════════════

def train_subject(
    subject_id: int,
    data_root_dict: Dict[str, str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """
    训练单个被试的 FitNets 模型
    
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

    # ── 训练 Student (FitNets) ────────────────────────────────────────────────
    print(f"  Training Student (FitNets)...")
    
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
    
    # regressor：student eeg_feat [B,128] → teacher fused [B,384]
    regressor = HintRegressor(
        student_dim = STUDENT_DIM,   # 128
        teacher_dim = TEACHER_DIM,   # 384
    ).to(device)

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

    print(f"  Stage 1 — Hint pre-training")

    best_hint_loss = float('inf')
    hint_patience  = 0
    hint_ckpt      = os.path.join(SAVE_DIR, f'best_hint_regressor_s{subject_id:02d}.pth')

    for epoch in range(1, EPOCHS_STAGE1 + 1):
        tr_loss, _, _ = run_epoch(1, student, teacher, regressor,
                                  train_loader, opt1, device, epoch, is_train=True)
        va_loss, _, _ = run_epoch(1, student, teacher, regressor,
                                  test_loader,   opt1, device, epoch, is_train=False)

        if va_loss < best_hint_loss:
            best_hint_loss = va_loss
            hint_patience  = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save({
                'epoch'          : epoch,
                'student_encoder': {
                    'sgcn'   : student.sgcn.state_dict(),
                    'abilstm': student.abilstm.state_dict(),
                },
                'regressor'      : regressor.state_dict(),
                'best_hint_loss' : best_hint_loss,
            }, hint_ckpt)
        else:
            hint_patience += 1

        if hint_patience >= PATIENCE:
            break

    # 加载 Stage 1 最优编码器权重，作为 Stage 2 初始化
    best_s1 = torch.load(hint_ckpt, map_location=device)
    student.sgcn.load_state_dict(best_s1['student_encoder']['sgcn'])
    student.abilstm.load_state_dict(best_s1['student_encoder']['abilstm'])

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
    ckpt_path      = os.path.join(SAVE_DIR, f'best_student_fitnets_s{subject_id:02d}.pth')

    print(f"  Stage 2 — Full KD fine-tuning")

    for epoch in range(1, EPOCHS_STAGE2 + 1):
        tr_loss, tr_acc, tr_f1 = run_epoch(2, student, teacher, regressor,
                                           train_loader, opt2, device, epoch, is_train=True)
        va_loss, va_acc, va_f1 = run_epoch(2, student, teacher, regressor,
                                           test_loader,   opt2, device, epoch, is_train=False)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            torch.save({'epoch'       : epoch,
                        'model_state' : student.state_dict(),
                        'best_val_acc': best_val_acc}, ckpt_path)
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    student.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
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
