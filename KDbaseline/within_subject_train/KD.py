import os
import json
import csv
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# ── 项目内模块 ─────────────────────────────────────────────────────────────────
from KDbaseline.model.Teacher import TeacherModel, TeacherLoss
from KDbaseline.model.Student import ST_GCLSTM
from dataset.within_subject_dataset import create_within_subject_dataloaders, create_within_subject_dataloaders_multimodal


# =============================================================================
# 结果记录器
# =============================================================================

class ResultRecorder:
    """
    记录每个被试的指标，支持导出 CSV / JSON。

    字段：
        subject_id, teacher_acc, teacher_f1_macro, teacher_f1_weighted,
        student_acc, student_f1_macro, student_f1_weighted,
        train_samples, test_samples,
        best_epoch_teacher, best_epoch_student, train_time_s
    """

    FIELDS = [
        'subject_id', 'teacher_acc', 'teacher_f1_macro', 'teacher_f1_weighted',
        'student_acc', 'student_f1_macro', 'student_f1_weighted',
        'train_samples', 'test_samples',
        'best_epoch_teacher', 'best_epoch_student', 'train_time_s',
    ]

    def __init__(self, save_dir: str, exp_name: str = 'within_subject_kd'):
        self.save_dir = save_dir
        self.exp_name = exp_name
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path  = os.path.join(save_dir, f'{exp_name}_{ts}.csv')
        self.json_path = os.path.join(save_dir, f'{exp_name}_{ts}.json')

        self.records: List[Dict[str, Any]] = []

        # 写 CSV 表头
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()

        print(f"📝 Results will be saved to:\n"
              f"   CSV  → {self.csv_path}\n"
              f"   JSON → {self.json_path}")

    # ── 添加单个被试结果 ────────────────────────────────────────────────────────

    def add(self,
            subject_id:    int,
            y_true:        np.ndarray,
            teacher_pred:  np.ndarray,
            student_pred:  np.ndarray,
            train_samples: int  = 0,
            test_samples:  int  = 0,
            best_epoch_teacher: int  = 0,
            best_epoch_student: int  = 0,
            train_time_s:  float = 0.0):

        # Teacher 指标
        teacher_acc         = accuracy_score(y_true, teacher_pred)
        teacher_f1_macro    = f1_score(y_true, teacher_pred, average='macro',    zero_division=0)
        teacher_f1_weighted = f1_score(y_true, teacher_pred, average='weighted', zero_division=0)

        # Student 指标
        student_acc         = accuracy_score(y_true, student_pred)
        student_f1_macro    = f1_score(y_true, student_pred, average='macro',    zero_division=0)
        student_f1_weighted = f1_score(y_true, student_pred, average='weighted', zero_division=0)

        record = {
            'subject_id':          subject_id,
            'teacher_acc':         round(teacher_acc,         4),
            'teacher_f1_macro':    round(teacher_f1_macro,    4),
            'teacher_f1_weighted': round(teacher_f1_weighted, 4),
            'student_acc':         round(student_acc,         4),
            'student_f1_macro':    round(student_f1_macro,    4),
            'student_f1_weighted': round(student_f1_weighted, 4),
            'train_samples':       train_samples,
            'test_samples':        test_samples,
            'best_epoch_teacher':  best_epoch_teacher,
            'best_epoch_student':  best_epoch_student,
            'train_time_s':        round(train_time_s, 2),
        }
        self.records.append(record)

        # 实时追加到 CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(record)

        print(f"  [S{subject_id:02d}] Teacher: Acc={teacher_acc:.4f} F1={teacher_f1_weighted:.4f} | "
              f"Student: Acc={student_acc:.4f} F1={student_f1_weighted:.4f} | "
              f"(t={train_time_s:.1f}s)")
        return record

    # ── 保存汇总 + 打印表格 ─────────────────────────────────────────────────────

    def save_summary(self) -> Dict[str, float]:
        if not self.records:
            print("⚠️  No records to summarize.")
            return {}

        # Teacher 指标
        teacher_accs         = [r['teacher_acc']         for r in self.records]
        teacher_f1_macros    = [r['teacher_f1_macro']    for r in self.records]
        teacher_f1_weighteds = [r['teacher_f1_weighted'] for r in self.records]

        # Student 指标
        student_accs         = [r['student_acc']         for r in self.records]
        student_f1_macros    = [r['student_f1_macro']    for r in self.records]
        student_f1_weighteds = [r['student_f1_weighted'] for r in self.records]

        summary = {
            'num_subjects':              len(self.records),
            # Teacher
            'teacher_mean_acc':          round(float(np.mean(teacher_accs)),         4),
            'teacher_std_acc':           round(float(np.std(teacher_accs)),          4),
            'teacher_mean_f1_macro':     round(float(np.mean(teacher_f1_macros)),    4),
            'teacher_std_f1_macro':      round(float(np.std(teacher_f1_macros)),     4),
            'teacher_mean_f1_weighted':  round(float(np.mean(teacher_f1_weighteds)), 4),
            'teacher_std_f1_weighted':   round(float(np.std(teacher_f1_weighteds)),  4),
            # Student
            'student_mean_acc':          round(float(np.mean(student_accs)),         4),
            'student_std_acc':           round(float(np.std(student_accs)),          4),
            'student_mean_f1_macro':     round(float(np.mean(student_f1_macros)),    4),
            'student_std_f1_macro':      round(float(np.std(student_f1_macros)),     4),
            'student_mean_f1_weighted':  round(float(np.mean(student_f1_weighteds)), 4),
            'student_std_f1_weighted':   round(float(np.std(student_f1_weighteds)),  4),
        }

        # JSON 写入（records + summary）
        output = {
            'experiment': self.exp_name,
            'timestamp':  datetime.now().isoformat(),
            'summary':    summary,
            'per_subject': self.records,
        }
        with open(self.json_path, 'w') as f:
            json.dump(output, f, indent=2)

        # 追加 summary 行到 CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(['=== SUMMARY ==='])
            for k, v in summary.items():
                writer.writerow([k, v])

        self._print_table(summary)
        return summary

    # ── 控制台表格 ──────────────────────────────────────────────────────────────

    def _print_table(self, summary: Dict):
        sep = '─' * 95
        print(f"\n{'═'*95}")
        print(f"  {'Subject':>8}  {'Teacher Acc':>10}  {'Teacher F1':>10}  {'Student Acc':>10}  {'Student F1':>10}")
        print(sep)
        for r in self.records:
            print(f"  S{r['subject_id']:02d}      "
                  f"{r['teacher_acc']:>10.4f}  "
                  f"{r['teacher_f1_weighted']:>10.4f}  "
                  f"{r['student_acc']:>10.4f}  "
                  f"{r['student_f1_weighted']:>10.4f}")
        print(sep)
        print(f"  {'Mean':>8}  "
              f"{summary['teacher_mean_acc']:>10.4f}  "
              f"{summary['teacher_mean_f1_weighted']:>10.4f}  "
              f"{summary['student_mean_acc']:>10.4f}  "
              f"{summary['student_mean_f1_weighted']:>10.4f}")
        print(f"  {'Std':>8}  "
              f"{summary['teacher_std_acc']:>10.4f}  "
              f"{summary['teacher_std_f1_weighted']:>10.4f}  "
              f"{summary['student_std_acc']:>10.4f}  "
              f"{summary['student_std_f1_weighted']:>10.4f}")
        print('═' * 95)
        print(f"  📄 CSV  → {self.csv_path}")
        print(f"  📄 JSON → {self.json_path}\n")


# =============================================================================
# KD Loss
# =============================================================================

class KDLoss(nn.Module):
    """
    L = w_ce * CE(s_logits, labels)
      + w_kd * T^2 * KL(softmax(s/T) || softmax(t/T))
    """
    def __init__(self, temperature: float = 4.0,
                 w_ce: float = 1.0, w_kd: float = 1.0):
        super().__init__()
        self.T    = temperature
        self.w_ce = w_ce
        self.w_kd = w_kd
        self.ce   = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, labels):
        l_ce = self.ce(s_logits, labels)

        s_log = torch.nn.functional.log_softmax(s_logits / self.T, dim=-1)
        t_soft = torch.nn.functional.softmax(t_logits.detach() / self.T, dim=-1)
        l_kd  = torch.nn.functional.kl_div(s_log, t_soft, reduction='batchmean') * (self.T ** 2)

        total = self.w_ce * l_ce + self.w_kd * l_kd
        return {'loss': total, 'l_ce': l_ce, 'l_kd': l_kd}


# =============================================================================
# 工具
# =============================================================================

def set_seed(seed):
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
# 多模态 Dataset
# =============================================================================

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


# =============================================================================
# 训练 / 评估工具函数
# =============================================================================

def train_teacher_one_epoch(model:     nn.Module,
                            loader:    DataLoader,
                            optimizer: optim.Optimizer,
                            criterion: nn.Module,
                            device:    torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        eeg, pcc, audio, vision, y = batch
        eeg, pcc, audio, vision, y = [d.to(device) for d in (eeg, pcc, audio, vision, y)]
        
        out = model(eeg, pcc, audio, vision)
        loss = criterion(out, y)['loss']
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_teacher(model:  nn.Module,
                     loader: DataLoader,
                     device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            eeg, pcc, audio, vision, y = batch
            eeg, pcc, audio, vision = [d.to(device) for d in (eeg, pcc, audio, vision)]
            
            out = model(eeg, pcc, audio, vision)
            preds = out['logits'].argmax(dim=-1).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(y.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def train_student_one_epoch(teacher:   nn.Module,
                            student:   nn.Module,
                            loader:    DataLoader,
                            criterion: nn.Module,
                            optimizer: optim.Optimizer,
                            device:    torch.device) -> Dict[str, float]:
    student.train()
    teacher.eval()
    
    meters = {k: AverageMeter() for k in ('loss', 'l_ce', 'l_kd')}
    all_preds, all_labels = [], []
    
    with torch.enable_grad():
        for batch in loader:
            eeg, pcc, audio, vision, y = batch
            eeg, pcc, audio, vision, y = [d.to(device) for d in (eeg, pcc, audio, vision, y)]
            
            # Teacher 推理
            with torch.no_grad():
                t_out = teacher(eeg, pcc, audio, vision)
            
            # Student 推理
            s_out = student(eeg, pcc)
            
            # 计算损失
            losses = criterion(s_out['logits'], t_out['logits'], y)
            
            # 反向传播
            optimizer.zero_grad()
            losses['loss'].backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            # 更新指标
            batch_size = y.size(0)
            for k in meters:
                meters[k].update(losses[k].item(), batch_size)
            
            all_preds.extend(s_out['logits'].argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'acc': acc,
        'f1': f1,
        'loss': meters['loss'].avg,
        'l_ce': meters['l_ce'].avg,
        'l_kd': meters['l_kd'].avg,
    }

def evaluate_student(teacher:   nn.Module,
                     student:   nn.Module,
                     loader:    DataLoader,
                     criterion: nn.Module,
                     device:    torch.device) -> Tuple[np.ndarray, np.ndarray]:
    student.eval()
    teacher.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            eeg, pcc, audio, vision, y = batch
            eeg, pcc, audio, vision = [d.to(device) for d in (eeg, pcc, audio, vision)]
            
            # Student 推理
            s_out = student(eeg, pcc)
            preds = s_out['logits'].argmax(dim=-1).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(y.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# =============================================================================
# 单被试训练循环
# =============================================================================

def train_subject(subject_id:  int,
                  data_root_dict: Dict[str, str],
                  num_epochs:   int   = 50,
                  lr:           float = 1e-3,
                  weight_decay: float = 1e-4,
                  patience:     int   = 10,
                  temperature:  float = 4.0,
                  w_ce:         float = 1.0,
                  w_kd:         float = 1.0,
                  device:       torch.device = torch.device('cpu'),
                  verbose:      bool  = False,
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """
    返回 (y_true, teacher_pred, student_pred, best_epoch_teacher, best_epoch_student, elapsed_seconds)
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
    
    # 初始化 Teacher 模型
    teacher = TeacherModel(
        audio_input_dim  = 25,
        vision_input_dim = 161,
        av_hidden_dim    = 64,
        av_num_layers    = 1,
        av_dropout       = 0.5,
        num_nodes        = 30,
        eeg_in_features  = 5,
        gcn_hidden       = 64,
        gcn_out          = 64,
        lstm_hidden      = 64,
        lstm_layers      = 1,
        eeg_dropout      = 0.5,
        fc_hidden        = 64,
        num_classes      = 5,
    ).to(device)

    teacher_optimizer = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    teacher_criterion = TeacherLoss()
    teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, T_max=num_epochs)

    best_teacher_acc       = -1.0
    best_teacher_state     = None
    best_epoch_teacher     = 0
    no_improve_teacher     = 0

    for epoch in range(1, num_epochs + 1):
        loss = train_teacher_one_epoch(teacher, train_loader, teacher_optimizer, teacher_criterion, device)
        teacher_scheduler.step()

        y_true, teacher_pred = evaluate_teacher(teacher, test_loader, device)
        acc = accuracy_score(y_true, teacher_pred)

        if acc > best_teacher_acc:
            best_teacher_acc   = acc
            best_teacher_state = {k: v.clone() for k, v in teacher.state_dict().items()}
            best_epoch_teacher = epoch
            no_improve_teacher = 0
        else:
            no_improve_teacher += 1

        if verbose and epoch % 10 == 0:
            print(f"    Teacher epoch {epoch:3d}/{num_epochs}  loss={loss:.4f}  "
                  f"acc={acc:.4f}  best={best_teacher_acc:.4f}")

        if no_improve_teacher >= patience:
            if verbose:
                print(f"    Teacher early stop at epoch {epoch}.")
            break

    # 加载最佳 Teacher 权重
    teacher.load_state_dict(best_teacher_state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── 训练 Student (KD) ────────────────────────────────────────────────────
    print(f"  Training Student (KD)...")
    
    # 初始化 Student 模型
    student = ST_GCLSTM(
        num_nodes   = 30,
        in_features = 5,
        gcn_hidden  = 64,
        gcn_out     = 64,
        lstm_hidden = 64,
        lstm_layers = 1,
        fc_hidden   = 64,
        num_classes = 5,
        dropout     = 0.5,
    ).to(device)

    student_optimizer = optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    student_criterion = KDLoss(temperature=temperature, w_ce=w_ce, w_kd=w_kd)
    student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(student_optimizer, mode='max', factor=0.5, patience=7)

    best_student_acc       = -1.0
    best_student_state     = None
    best_epoch_student     = 0
    no_improve_student     = 0

    for epoch in range(1, num_epochs + 1):
        tr_result = train_student_one_epoch(teacher, student, train_loader, student_criterion, student_optimizer, device)
        
        # 评估
        y_true, student_pred = evaluate_student(teacher, student, test_loader, student_criterion, device)
        acc = accuracy_score(y_true, student_pred)
        
        student_scheduler.step(acc)

        if acc > best_student_acc:
            best_student_acc   = acc
            best_student_state = {k: v.clone() for k, v in student.state_dict().items()}
            best_epoch_student = epoch
            no_improve_student = 0
        else:
            no_improve_student += 1

        if verbose and epoch % 10 == 0:
            print(f"    Student epoch {epoch:3d}/{num_epochs}  loss={tr_result['loss']:.4f}  "
                  f"acc={acc:.4f}  best={best_student_acc:.4f}")

        if no_improve_student >= patience:
            if verbose:
                print(f"    Student early stop at epoch {epoch}.")
            break

    # 加载最佳 Student 权重
    student.load_state_dict(best_student_state)
    student.eval()

    # 最终评估
    y_true, teacher_pred = evaluate_teacher(teacher, test_loader, device)
    y_true, student_pred = evaluate_student(teacher, student, test_loader, student_criterion, device)

    elapsed = time.time() - t0
    return y_true, teacher_pred, student_pred, best_epoch_teacher, best_epoch_student, elapsed


# =============================================================================
# 主流程：遍历所有被试
# =============================================================================

def run_within_subject_experiment(
        data_root_dict:      Dict[str, str],
        subject_ids:         List[int],
        save_dir:            str   = './results',
        exp_name:            str   = 'within_subject_kd',
        batch_size:          int   = 64,
        num_epochs:          int   = 50,
        lr:                  float = 1e-3,
        weight_decay:        float = 1e-4,
        patience:            int   = 10,
        temperature:         float = 4.0,
        w_ce:                float = 1.0,
        w_kd:                float = 1.0,
        device:              Optional[torch.device] = None,
        verbose:             bool  = False,
) -> Dict[str, Any]:

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    recorder = ResultRecorder(save_dir=save_dir, exp_name=exp_name)

    for sid in subject_ids:
        print(f"\n{'─'*50}")
        print(f"🔄 Subject {sid:02d} / {max(subject_ids)}")

        # 训练
        y_true, teacher_pred, student_pred, best_epoch_teacher, best_epoch_student, elapsed = train_subject(
            subject_id     = sid,
            data_root_dict = data_root_dict,
            num_epochs     = num_epochs,
            lr             = lr,
            weight_decay   = weight_decay,
            patience       = patience,
            temperature    = temperature,
            w_ce           = w_ce,
            w_kd           = w_kd,
            device         = device,
            verbose        = verbose,
        )

        # 记录
        recorder.add(
            subject_id          = sid,
            y_true              = y_true,
            teacher_pred        = teacher_pred,
            student_pred        = student_pred,
            train_samples       = len(train_loader.dataset) if 'train_loader' in locals() else 0,
            test_samples        = len(test_loader.dataset) if 'test_loader' in locals() else 0,
            best_epoch_teacher  = best_epoch_teacher,
            best_epoch_student  = best_epoch_student,
            train_time_s        = elapsed,
        )

        # 详细分类报告（可选）
        if verbose:
            print("\nTeacher Classification Report:")
            print(classification_report(y_true, teacher_pred, zero_division=0))
            print("\nStudent Classification Report:")
            print(classification_report(y_true, student_pred, zero_division=0))

    summary = recorder.save_summary()
    return {'summary': summary, 'records': recorder.records}


# =============================================================================
# CLI 入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Within-Subject KD Experiment')
    parser.add_argument('--eeg_root',    type=str, default='/data2/mingzhi/BCI/dataset/EAV_old/EEG')
    parser.add_argument('--audio_root',  type=str, default='/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw')
    parser.add_argument('--vision_root', type=str, default='/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace')
    parser.add_argument('--save_dir',    type=str, default='./results')
    parser.add_argument('--exp_name',    type=str, default='within_subject_kd')
    parser.add_argument('--num_subjects',type=int, default=42)
    parser.add_argument('--subjects',    type=int, nargs='+', default=None,
                        help='指定被试 ID，默认全部')
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--patience',    type=int,   default=10)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--w_ce',        type=float, default=1.0)
    parser.add_argument('--w_kd',        type=float, default=1.0)
    parser.add_argument('--verbose',     action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    
    # 构建数据根目录字典
    data_root_dict = {
        'eeg'   : args.eeg_root,
        'audio' : args.audio_root,
        'vision': args.vision_root,
    }
    
    # 确定被试列表
    subjects = args.subjects or list(range(1, args.num_subjects + 1))
    
    # 运行实验
    run_within_subject_experiment(
        data_root_dict = data_root_dict,
        subject_ids    = subjects,
        save_dir       = args.save_dir,
        exp_name       = args.exp_name,
        batch_size     = args.batch_size,
        num_epochs     = args.epochs,
        lr             = args.lr,
        patience       = args.patience,
        temperature    = args.temperature,
        w_ce           = args.w_ce,
        w_kd           = args.w_kd,
        verbose        = args.verbose,
    )
