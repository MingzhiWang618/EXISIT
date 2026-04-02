import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional, Dict, List, Tuple

# 加入路径以便加载 dataset
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# ── 项目内模块 ─────────────────────────────────────────────────────────────────
from KDbaseline.model.Teacher import TeacherModel   # 复用作 TA
from KDbaseline.model.Student import ST_GCLSTM
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

    # ── 模型（EEG 分支，Teacher / TA / Student 共享）─────────────────────────
    num_nodes   = 30
    in_features = 5
    gcn_hidden  = 64
    gcn_out     = 64
    lstm_hidden = 64
    lstm_layers = 1
    fc_hidden   = 64
    num_classes = 5
    dropout     = 0.5

    # ── AV 分支容量逐级缩减（Teacher → TA1 → TA2）───────────────────────────
    audio_dim      = 25
    vision_dim     = 161
    av_hidden_teacher = 64    # Teacher：完整 AV
    av_hidden_ta1     = 32    # TA1：减半
    av_hidden_ta2     = 16    # TA2：再减半

    # ── 训练 ──────────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 1e-3
    weight_decay = 1e-3
    patience     = 20
    seed         = 2024
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── CDGKD 超参（论文 β=0.5）──────────────────────────────────────────────
    beta        = 0.5    # CE 与 KD loss 的权重
    temperature = 4.0    # 软标签温度

    # ── 路径 ──────────────────────────────────────────────────────────────────
    save_dir      = './checkpoints'
    teacher_ckpt  = './checkpoints/best_teacher_proto.pth'
    ta1_ckpt      = './checkpoints/best_ta1.pth'
    ta2_ckpt      = './checkpoints/best_ta2.pth'
    student_ckpt  = './checkpoints/best_student_cdgkd.pth'


# =============================================================================
# CDGKD Loss（论文公式 9 + 随机学习策略）
# =============================================================================

class CDGKDLoss(nn.Module):
    """
    L = (1-β)·L_CE(curr_logits, labels)
      + β · (1/|S_i|) · Σ_{j∈S_i} KL(curr_logits/T ‖ upper_logits_j/T) · T²

    随机学习策略：从上级网络 logits 列表中随机采样子集 S_i。
    训练时 stochastic=True，验证/测试时 stochastic=False（用全部上级）。
    """
    def __init__(self, beta: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.beta = beta
        self.T    = temperature
        self.ce   = nn.CrossEntropyLoss()

    def forward(self,
                curr_logits: torch.Tensor,
                upper_logits_list: list,
                labels: torch.Tensor,
                stochastic: bool = True) -> dict:
        """
        curr_logits       : [B, C]  当前网络（TA_i 或 Student）的 logits
        upper_logits_list : list of [B, C]，所有上级网络的 logits
        labels            : [B]
        stochastic        : 是否随机采样子集（训练时 True，验证时 False）
        """
        l_ce = self.ce(curr_logits, labels)

        # 随机学习策略：从上级 logits 中随机采样非空子集
        if stochastic and len(upper_logits_list) > 1:
            k      = random.randint(1, len(upper_logits_list))
            subset = random.sample(upper_logits_list, k)
        else:
            subset = upper_logits_list

        # KL divergence 对每个上级求平均
        kl_losses = []
        s_log = F.log_softmax(curr_logits / self.T, dim=-1)
        for t_logits in subset:
            t_soft = F.softmax(t_logits.detach() / self.T, dim=-1)
            kl     = F.kl_div(s_log, t_soft, reduction='batchmean') * (self.T ** 2)
            kl_losses.append(kl)

        l_kd = torch.stack(kl_losses).mean()
        loss = (1 - self.beta) * l_ce + self.beta * l_kd

        return {'loss': loss, 'l_ce': l_ce, 'l_kd': l_kd}


# =============================================================================
# 工具
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def load_model(model, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        acc = ckpt.get('best_val_acc', ckpt.get('val_acc', '?'))
        print(f"  Loaded {ckpt_path}  (val_acc={acc})")
    return model


# =============================================================================
# 单 epoch：TA 或 Student 训练/验证
# =============================================================================

def run_epoch(cfg, current_net, upper_nets, loader,
              criterion, optimizer=None, epoch=0,
              current_uses_av=True):
    """
    current_net     : 当前训练的网络（TA 或 Student）
    upper_nets      : list of 已固定的上级网络（Teacher, TA1, TA2, ...）
    current_uses_av : 当前网络是否需要 audio/vision 输入（Student 不需要）
    """
    is_train   = optimizer is not None
    stochastic = is_train     # 只在训练时启用随机学习策略
    current_net.train() if is_train else current_net.eval()

    meters = {k: AverageMeter() for k in ('loss', 'l_ce', 'l_kd')}
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for eeg, pcc, audio, vision, y in loader:
            eeg, pcc, audio, vision, y = [
                d.to(cfg.device) for d in (eeg, pcc, audio, vision, y)]

            # ── 上级网络推理（全部冻结，无梯度）────────────────────────────
            upper_logits_list = []
            with torch.no_grad():
                for net in upper_nets:
                    # upper_nets 全部是 TeacherModel 结构（有 AV 输入）
                    out = net(eeg, pcc, audio, vision)
                    upper_logits_list.append(out['logits'])

            # ── 当前网络前向 ────────────────────────────────────────────────
            if current_uses_av:
                # TA：TeacherModel 结构，需要 AV 输入
                if is_train:
                    curr_out = current_net(eeg, pcc, audio, vision)
                else:
                    with torch.no_grad():
                        curr_out = current_net(eeg, pcc, audio, vision)
            else:
                # Student：ST_GCLSTM，只用 EEG
                if is_train:
                    curr_out = current_net(eeg, pcc)
                else:
                    with torch.no_grad():
                        curr_out = current_net(eeg, pcc)

            curr_logits = curr_out['logits']

            # ── 损失计算 ────────────────────────────────────────────────────
            losses = criterion(curr_logits, upper_logits_list, y, stochastic)

            if is_train:
                optimizer.zero_grad()
                losses['loss'].backward()
                nn.utils.clip_grad_norm_(current_net.parameters(), 1.0)
                optimizer.step()

            n = eeg.size(0)
            for k in meters:
                meters[k].update(losses[k].item(), n)
            all_preds.extend(curr_logits.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'acc' : acc, 'f1': f1,
        'loss': meters['loss'].avg,
        'l_ce': meters['l_ce'].avg,
        'l_kd': meters['l_kd'].avg,
    }


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# =============================================================================
# 单个网络的完整训练流程（含 early stopping）
# =============================================================================

def train_one_level(cfg, name, current_net, upper_nets,
                    train_loader, val_loader,
                    criterion, ckpt_path,
                    current_uses_av=True):

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, current_net.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    best_val_acc   = 0.0
    patience_count = 0

    print(f"  Training {name}...")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr = run_epoch(cfg, current_net, upper_nets, train_loader,
                       criterion, optimizer, epoch, current_uses_av)
        va = run_epoch(cfg, current_net, upper_nets, val_loader,
                       criterion, epoch=epoch,
                       current_uses_av=current_uses_av)

        scheduler.step(va['acc'])
        lr_now  = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        if va['acc'] > best_val_acc:
            best_val_acc   = va['acc']
            patience_count = 0
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({'epoch'       : epoch,
                        'model_state' : current_net.state_dict(),
                        'best_val_acc': best_val_acc}, ckpt_path)
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    print(f"  Best val acc ({name}): {best_val_acc:.4f}")
    # 加载最优权重供下一级使用
    current_net.load_state_dict(
        torch.load(ckpt_path, map_location=cfg.device)['model_state'])
    return current_net, best_val_acc


# ═══════════════════════════════════════════════════════════════════════════════
# 训练单个被试
# ═══════════════════════════════════════════════════════════════════════════════

def train_subject(
    subject_id: int,
    data_root_dict: Dict[str, str],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    """
    训练单个被试的 CDGKD 模型
    
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

    # ── 配置 ──────────────────────────────────────────────────────────────────
    cfg = Config()
    set_seed(cfg.seed)

    # ── CDGKD Loss ────────────────────────────────────────────────────────────
    criterion = CDGKDLoss(beta=cfg.beta, temperature=cfg.temperature)

    # =========================================================================
    # Step 0：训练 Teacher（如果没有预训练权重）
    # =========================================================================
    print("  Training Teacher...")
    teacher = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden_teacher,
        av_num_layers    = cfg.lstm_layers,
        av_dropout       = cfg.dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)

    # 训练 Teacher
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    teacher_criterion = nn.CrossEntropyLoss()
    
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
            loss = teacher_criterion(outputs['logits'], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
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
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save({
                'model_state': teacher.state_dict(),
                'best_val_acc': best_teacher_acc,
            }, cfg.teacher_ckpt)
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Train Loss={total_loss/total:.4f}, Val Acc={val_acc:.4f}")
    
    freeze(teacher)

    # =========================================================================
    # Step 1：训练 TA1  ← {Teacher}
    # 论文公式 7：A1 ← {T}
    # =========================================================================
    ta1 = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden_ta1,   # 32，AV 减半
        av_num_layers    = cfg.lstm_layers,
        av_dropout       = cfg.dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)

    ta1, _ = train_one_level(
        cfg, name='TA1',
        current_net   = ta1,
        upper_nets    = [teacher],           # 论文公式 7：A1 ← {T}
        train_loader  = train_loader,
        val_loader    = test_loader,
        criterion     = criterion,
        ckpt_path     = os.path.join(cfg.save_dir, f'best_ta1_s{subject_id:02d}.pth'),
        current_uses_av = True,
    )
    freeze(ta1)

    # =========================================================================
    # Step 2：训练 TA2  ← {Teacher, TA1}
    # 论文公式 7：A2 ← {T, A1}，随机采样子集
    # =========================================================================
    ta2 = TeacherModel(
        audio_input_dim  = cfg.audio_dim,
        vision_input_dim = cfg.vision_dim,
        av_hidden_dim    = cfg.av_hidden_ta2,   # 16，再减半
        av_num_layers    = cfg.lstm_layers,
        av_dropout       = cfg.dropout,
        num_nodes        = cfg.num_nodes,
        eeg_in_features  = cfg.in_features,
        gcn_hidden       = cfg.gcn_hidden,
        gcn_out          = cfg.gcn_out,
        lstm_hidden      = cfg.lstm_hidden,
        lstm_layers      = cfg.lstm_layers,
        eeg_dropout      = cfg.dropout,
        fc_hidden        = cfg.fc_hidden,
        num_classes      = cfg.num_classes,
    ).to(device)

    ta2, _ = train_one_level(
        cfg, name='TA2',
        current_net   = ta2,
        upper_nets    = [teacher, ta1],      # 论文公式 7：A2 ← {T, A1}
        train_loader  = train_loader,
        val_loader    = test_loader,
        criterion     = criterion,
        ckpt_path     = os.path.join(cfg.save_dir, f'best_ta2_s{subject_id:02d}.pth'),
        current_uses_av = True,
    )
    freeze(ta2)

    # =========================================================================
    # Step 3：训练 Student  ← {Teacher, TA1, TA2}
    # 论文公式 8：S ← {T} ∪ A，随机采样子集
    # =========================================================================
    student = ST_GCLSTM(
        num_nodes   = cfg.num_nodes,
        in_features = cfg.in_features,
        gcn_hidden  = cfg.gcn_hidden,
        gcn_out     = cfg.gcn_out,
        lstm_hidden = cfg.lstm_hidden,
        lstm_layers = cfg.lstm_layers,
        fc_hidden   = cfg.fc_hidden,
        num_classes = cfg.num_classes,
        dropout     = cfg.dropout,
    ).to(device)

    student, best_student_acc = train_one_level(
        cfg, name='Student',
        current_net   = student,
        upper_nets    = [teacher, ta1, ta2],  # 论文公式 8：S ← {T} ∪ A
        train_loader  = train_loader,
        val_loader    = test_loader,
        criterion     = criterion,
        ckpt_path     = os.path.join(cfg.save_dir, f'best_student_cdgkd_s{subject_id:02d}.pth'),
        current_uses_av = False,              # Student 只用 EEG
    )

    # ── 测试 ──────────────────────────────────────────────────────────────────
    student.load_state_dict(
        torch.load(os.path.join(cfg.save_dir, f'best_student_cdgkd_s{subject_id:02d}.pth'), map_location=device)['model_state'])
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
    cfg = Config()
    run_within_subject_experiment(
        data_root_dict=cfg.data_root_dict,
        subjects=list(range(1, 43)),
    )
