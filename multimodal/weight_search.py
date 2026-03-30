"""
search_distill_weights.py
─────────────────────────
用 Optuna (TPE 贝叶斯优化) 整夜搜索蒸馏损失权重的最优组合。

搜索空间：
    w_ce        : [0.5, 2.0]
    w_graph     : [1.0, 50.0]   log-uniform（量级跨度大）
    w_temporal  : [0.5, 10.0]   log-uniform
    w_logits    : [0.5, 5.0]
    temperature : {2, 3, 4, 6, 8}

每个 trial 训练固定 TRIAL_EPOCHS 轮（快速代理），
以最佳 val_acc 作为优化目标。
全部 trial 结束后输出 Top-10 结果并保存到 CSV。

用法：
    pip install optuna
    python search_distill_weights.py
"""

import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

import csv
import json
import time
import logging
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler

from dataset.dataset import CrossSubjectMultiModalDataset
from multimodal.model.Teacher import TeacherModel
from multimodal.model.Student import ST_GCLSTM

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# 固定超参（不参与搜索）
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
DK           = 32

TRIAL_EPOCHS = 30       # 每个 trial 的训练轮数（代理评估，不需要收敛）
N_TRIALS     = 120      # 总 trial 数（~整夜）
PATIENCE     = 10       # trial 内 early stopping
SAVE_DIR     = './search_ckpts'
RESULT_CSV   = './distill_search_results.csv'
STUDY_DB     = 'sqlite:///distill_search.db'   # 断点续跑
DEVICE       = 'cuda'


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalDataset(Dataset):
    def __init__(self, eeg, pcc, audio, vision, labels,
                 vision_max_len=None):
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


# ═══════════════════════════════════════════════════════════════════════════════
# 数据（全局加载一次，所有 trial 复用）
# ═══════════════════════════════════════════════════════════════════════════════

def build_loaders(vision_max_len=None):
    manager = CrossSubjectMultiModalDataset(
        DATA_ROOT,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )
    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True,
                               normalize=True, compute_pcc=True)
    (tr_aud, _), (va_aud, _), (te_aud, _) = \
        manager.get_all_splits('audio', extract_de=False, normalize=True)
    (tr_vis, _), (va_vis, _), (te_vis, _) = \
        manager.get_all_splits('vision', extract_de=False, normalize=True)

    def make(eeg, pcc, aud, vis, y, shuffle):
        ds = MultiModalDataset(eeg, pcc, aud, vis, y, vision_max_len)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True, drop_last=False)

    return (make(tr_eeg, tr_pcc, tr_aud, tr_vis, tr_y, True),
            make(va_eeg, va_pcc, va_aud, va_vis, va_y, False),
            make(te_eeg, te_pcc, te_aud, te_vis, te_y, False))


# ═══════════════════════════════════════════════════════════════════════════════
# 蒸馏损失
# ═══════════════════════════════════════════════════════════════════════════════

class DistillationLoss(nn.Module):
    def __init__(self, w_ce, w_graph, w_temporal, w_logits, temperature):
        super().__init__()
        self.w_ce       = w_ce
        self.w_graph    = w_graph
        self.w_temporal = w_temporal
        self.w_logits   = w_logits
        self.T          = temperature
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, s_out, t_out, labels):
        l_ce = self.ce(s_out['logits'], labels)

        # Graph KL：reduction=sum / B，对齐 CE 量纲（除以 B 而非 B*T*N）
        B, T_seq, N, _ = s_out['R'].shape
        s_R = s_out['R'].reshape(B * T_seq * N, N)
        t_R = t_out['R'].reshape(B * T_seq * N, N).detach()
        l_graph = F.kl_div(
            s_R.clamp(min=1e-8).log(),
            t_R.clamp(min=1e-8),
            reduction='sum',
        ) / B

        # Temporal KL
        s_attn = s_out['attn_t']
        t_attn = t_out['attn_t'].detach()
        l_temporal = F.kl_div(
            s_attn.clamp(min=1e-8).log(),
            t_attn.clamp(min=1e-8),
            reduction='sum',
        ) / B

        # Soft-logits KL
        t_soft     = F.softmax(t_out['eeg_logits'] / self.T, dim=-1).detach()
        s_log_soft = F.log_softmax(s_out['logits']  / self.T, dim=-1)
        l_logits   = F.kl_div(s_log_soft, t_soft,
                               reduction='batchmean') * (self.T ** 2)

        total = (self.w_ce       * l_ce
               + self.w_graph    * l_graph
               + self.w_temporal * l_temporal
               + self.w_logits   * l_logits)

        return {'loss': total, 'l_ce': l_ce, 'l_graph': l_graph,
                'l_temporal': l_temporal, 'l_logits': l_logits}


# ═══════════════════════════════════════════════════════════════════════════════
# 单 epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(teacher, student, loader, criterion,
              optimizer, device, is_train):
    teacher.eval()
    student.train() if is_train else student.eval()

    total = correct = 0
    sum_loss = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for eeg, pcc, audio, vision, labels in loader:
            eeg, pcc, audio, vision, labels = (
                eeg.to(device), pcc.to(device),
                audio.to(device), vision.to(device), labels.to(device))

            with torch.no_grad():
                t_out = teacher(eeg, pcc, audio, vision)

            s_out  = student(eeg, pcc)
            losses = criterion(s_out, t_out, labels)
            loss   = losses['loss']

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            B_        = labels.size(0)
            total    += B_
            sum_loss += loss.item() * B_
            correct  += (s_out['logits'].argmax(1) == labels).sum().item()

    return sum_loss / total, correct / total


# ═══════════════════════════════════════════════════════════════════════════════
# 单个 Trial（Optuna 目标函数）
# ═══════════════════════════════════════════════════════════════════════════════

def objective(trial, teacher, train_loader, val_loader, device,
              teacher_state):

    # ── 采样超参 ──────────────────────────────────────────────────────────────
    w_ce        = trial.suggest_float('w_ce',       0.5,  2.0)
    w_graph     = trial.suggest_float('w_graph',    1.0,  50.0, log=True)
    w_temporal  = trial.suggest_float('w_temporal', 0.5,  10.0, log=True)
    w_logits    = trial.suggest_float('w_logits',   0.5,  5.0)
    temperature = trial.suggest_categorical('temperature', [2, 3, 4, 6, 8])

    # ── 新建 Student（每个 trial 独立初始化）─────────────────────────────────
    student = ST_GCLSTM(
        num_nodes=NUM_NODES, in_features=IN_FEATURES,
        gcn_hidden=GCN_HIDDEN, gcn_out=GCN_OUT,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
        fc_hidden=FC_HIDDEN, num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    criterion = DistillationLoss(w_ce, w_graph, w_temporal,
                                 w_logits, temperature)
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    # ── 快速训练 TRIAL_EPOCHS 轮 ──────────────────────────────────────────────
    best_val   = 0.0
    no_improve = 0

    for epoch in range(1, TRIAL_EPOCHS + 1):
        run_epoch(teacher, student, train_loader, criterion,
                  optimizer, device, is_train=True)
        _, val_acc = run_epoch(teacher, student, val_loader, criterion,
                               None, device, is_train=False)
        scheduler.step(val_acc)

        # Optuna intermediate pruning（可选：剪掉明显差的 trial）
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_acc > best_val:
            best_val   = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_val


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")
    print(f"Trials  : {N_TRIALS}  ×  {TRIAL_EPOCHS} epochs each")

    # ── 数据（一次加载，所有 trial 共用）────────────────────────────────────
    print("\n📦 Loading data ...")
    train_loader, val_loader, test_loader = build_loaders()
    print(f"   train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    # ── Teacher（一次加载，冻结，所有 trial 共用）────────────────────────────
    teacher = TeacherModel(
        audio_input_dim=AUDIO_DIM, vision_input_dim=VISION_DIM,
        av_hidden_dim=AV_HIDDEN, av_num_layers=LSTM_LAYERS,
        av_dropout=DROPOUT, num_nodes=NUM_NODES,
        eeg_in_features=IN_FEATURES, gcn_hidden=GCN_HIDDEN,
        gcn_out=GCN_OUT, lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS, eeg_dropout=DROPOUT,
        dk=DK, fc_hidden=FC_HIDDEN, num_classes=NUM_CLASSES,
    ).to(device)

    teacher_ckpt = os.path.join(SAVE_DIR, '../checkpoints/best_teacher.pth')
    ckpt = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt['model_state'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_state = deepcopy(teacher.state_dict())  # 备份（实际不修改，保险起见）
    print(f"✅ Teacher loaded "
          f"(acc={ckpt.get('best_eeg_acc', ckpt.get('best_acc', '?'))})\n")

    # ── Optuna Study（支持断点续跑）─────────────────────────────────────────
    sampler = TPESampler(seed=42, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(
                  n_startup_trials=10,    # 前10个trial不剪枝
                  n_warmup_steps=10,      # 每个trial前10epoch不剪枝
                  interval_steps=2,
              )
    study = optuna.create_study(
        study_name   = 'distill_weight_search',
        storage      = STUDY_DB,
        direction    = 'maximize',
        sampler      = sampler,
        pruner       = pruner,
        load_if_exists = True,      # ← 断点续跑关键
    )

    already_done = len(study.trials)
    remaining    = N_TRIALS - already_done
    if remaining <= 0:
        print(f"Study already has {already_done} trials, nothing to do.")
    else:
        print(f"Starting {remaining} trials "
              f"(resuming from trial #{already_done}) ...\n")

    t0 = time.time()

    # 进度回调
    def progress_callback(study, trial):
        elapsed = (time.time() - t0) / 60
        done    = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
        best    = study.best_value if done > 0 else float('nan')
        print(f"  Trial {trial.number:>4d} │ "
              f"val_acc={trial.value:.4f} │ "
              f"best={best:.4f} │ "
              f"elapsed={elapsed:.1f}min │ "
              f"params={trial.params}")

    study.optimize(
        lambda trial: objective(
            trial, teacher, train_loader, val_loader,
            device, teacher_state),
        n_trials          = remaining,
        callbacks         = [progress_callback],
        catch             = (RuntimeError,),    # 跳过 OOM 等偶发错误
        show_progress_bar = False,
    )

    # ── 结果汇总 ──────────────────────────────────────────────────────────────
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    print(f"\n{'='*80}")
    print(f"  Search complete │ {len(completed)} completed trials")
    print(f"{'='*80}")
    print(f"  {'Rank':>4}  {'val_acc':>8}  "
          f"{'w_ce':>6}  {'w_graph':>8}  {'w_temp':>7}  "
          f"{'w_logits':>8}  {'T':>3}")
    print(f"  {'-'*70}")
    for rank, t in enumerate(completed[:10], 1):
        p = t.params
        print(f"  {rank:>4}  {t.value:>8.4f}  "
              f"{p['w_ce']:>6.3f}  {p['w_graph']:>8.3f}  "
              f"{p['w_temporal']:>7.3f}  "
              f"{p['w_logits']:>8.3f}  {p['temperature']:>3}")

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    fieldnames = ['rank', 'val_acc', 'w_ce', 'w_graph',
                  'w_temporal', 'w_logits', 'temperature',
                  'trial_number', 'duration_sec']
    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, t in enumerate(completed, 1):
            p = t.params
            writer.writerow({
                'rank'        : rank,
                'val_acc'     : round(t.value, 6),
                'w_ce'        : round(p['w_ce'],       4),
                'w_graph'     : round(p['w_graph'],     4),
                'w_temporal'  : round(p['w_temporal'],  4),
                'w_logits'    : round(p['w_logits'],    4),
                'temperature' : p['temperature'],
                'trial_number': t.number,
                'duration_sec': round(t.duration.total_seconds(), 1)
                                if t.duration else '',
            })
    print(f"\n✅ Full results saved → {RESULT_CSV}")

    # ── 最优参数保存为 JSON（方便直接复制到训练脚本）────────────────────────
    best_params = study.best_params
    best_json   = os.path.join(SAVE_DIR, 'best_distill_params.json')
    with open(best_json, 'w') as f:
        json.dump({**best_params, 'best_val_acc': study.best_value}, f, indent=2)
    print(f"✅ Best params saved  → {best_json}")
    print(f"\n🏆 Best config:")
    for k, v in best_params.items():
        print(f"   {k:<14} = {v}")
    print(f"   {'best_val_acc':<14} = {study.best_value:.4f}\n")


if __name__ == '__main__':
    main()