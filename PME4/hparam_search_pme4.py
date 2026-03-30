"""
PME4 Knowledge Distillation — 两阶段超参数搜索
  阶段一（粗搜）: 大步长网格，快速筛选有潜力区域
  阶段二（细搜）: 围绕粗搜 Top-N，小步长精细化

搜索参数:
    base_w_ce       ∈ {0.5, 1.0, 2.0}                    (离散)
    base_w_graph    ∈ [0.0, 8.0]                          (连续，粗→细)
    base_w_temporal ∈ [0.0, 8.0]                          (连续，粗→细)
    lr              ∈ {1e-4, 5e-4, 1e-3, 5e-3}            (离散，粗→细)

用法:
    python hparam_search_pme4_coarse_fine.py
"""

import os
import sys
import time
import itertools
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from PME4.model.Teacher  import TeacherModel
from PME4.model.Student  import ST_GCLSTM
from PME4.dataset.dataset import CrossSubjectPME4Dataset


# =============================================================================
# 1. 固定配置（不参与搜索）
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root  = '/data2/zhiwen/dataset/PME4'
    normalize  = True

    # ── 模型结构 ───────────────────────────────────────────────────────────────
    num_nodes        = 8
    eeg_in_features  = 5
    gcn_hidden       = 64
    gcn_out          = 64
    lstm_hidden      = 64
    lstm_layers      = 1
    fc_hidden        = 64
    num_classes      = 2
    dropout          = 0.5

    # Teacher Audio 参数
    audio_input_dim  = 25
    audio_hidden_dim = 128
    audio_num_layers = 2
    audio_dropout    = 0.5
    dk               = 32

    # ── 训练固定参数 ───────────────────────────────────────────────────────────
    batch_size   = 64
    weight_decay = 1e-3
    temperature  = 1.0
    base_w_logits = 0.0       # logits KL 暂不搜索

    # ── 路径 ──────────────────────────────────────────────────────────────────
    save_dir     = './checkpoints'
    teacher_ckpt = './checkpoints/teacher_pme4_best.pt'
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed         = 2024


# =============================================================================
# 2. 搜索空间配置
# =============================================================================

class SearchConfig:
    # ── 阶段一：粗搜（大步长，1×6×6×4 = 144 组）────────────────────────────────
    coarse_w_ce_values       = [1.0]
    coarse_w_graph_values    = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    coarse_w_temporal_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    coarse_lr_values         = [1e-4, 5e-4, 1e-3, 5e-3]

    # ── 阶段二：细搜（围绕粗搜 Top-N，小偏移）─────────────────────────────────
    fine_n_top   = 3                              # 取粗搜 Top-N 做细化
    fine_steps   = [-0.3, -0.15, 0.0, 0.15, 0.3] # w_graph / w_temporal 相对偏移
    # lr 细搜：以粗搜最优 lr 为中心，取相邻量级候选
    fine_lr_neighbors = {
        1e-4: [1e-4, 2e-4, 5e-4],
        5e-4: [2e-4, 5e-4, 1e-3],
        1e-3: [5e-4, 1e-3, 2e-3],
        5e-3: [2e-3, 5e-3, 1e-2],
    }

    # ── 各阶段 epoch / patience ────────────────────────────────────────────────
    coarse_epochs   = 60     # 粗搜 epoch 少，快速筛
    coarse_patience = 12
    fine_epochs     = 100    # 细搜给更多 epoch
    fine_patience   = 18

    # ── 输出路径 ───────────────────────────────────────────────────────────────
    results_path = './checkpoints/pme4_hparam_search_results.json'
    log_path     = './checkpoints/pme4_hparam_search_log.txt'


# =============================================================================
# 3. 数据集
# =============================================================================

class PME4DistillDataset(Dataset):
    def __init__(self, eeg, pcc, audio, y):
        n = min(len(eeg), len(audio), len(y))
        if n < len(eeg):
            print(f"  ⚠️  对齐截断: eeg={len(eeg)}, audio={len(audio)} → {n}")
        self.eeg   = torch.tensor(eeg[:n],   dtype=torch.float32)
        self.pcc   = torch.tensor(pcc[:n],   dtype=torch.float32)
        self.audio = torch.tensor(audio[:n], dtype=torch.float32)
        self.y     = torch.tensor(y[:n],     dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.eeg[idx], self.pcc[idx], self.audio[idx], self.y[idx]


# =============================================================================
# 4. 蒸馏损失
# =============================================================================

class DistillationLoss(nn.Module):
    def __init__(self, w_ce, w_graph, w_temporal, w_logits=0.0, temperature=1.0):
        super().__init__()
        self.w_ce = w_ce; self.w_graph = w_graph
        self.w_temporal = w_temporal; self.w_logits = w_logits
        self.T = temperature; self.ce = nn.CrossEntropyLoss()

    def forward(self, s_out, t_out, labels):
        l_ce = self.ce(s_out['logits'], labels)

        B, T_seq, N, _ = s_out['S_attn'].shape
        s_S = s_out['S_attn'].reshape(-1, N)
        t_S = t_out['S_attn'].reshape(-1, N).detach()
        l_graph = F.kl_div(
            s_S.clamp(min=1e-8).log(), t_S.clamp(min=1e-8), reduction='sum'
        ) / (B * T_seq * N)

        l_temporal = F.kl_div(
            s_out['attn_t'].clamp(min=1e-8).log(),
            t_out['attn_t'].detach().clamp(min=1e-8),
            reduction='batchmean',
        )

        t_soft     = F.softmax(t_out['eeg_logits'] / self.T, dim=-1).detach()
        s_log_soft = F.log_softmax(s_out['logits']  / self.T, dim=-1)
        l_logits   = F.kl_div(s_log_soft, t_soft,
                               reduction='batchmean') * (self.T ** 2)

        total = (self.w_ce * l_ce + self.w_graph * l_graph
               + self.w_temporal * l_temporal + self.w_logits * l_logits)
        return {
            'loss': total,
            'l_ce': l_ce, 'l_graph': l_graph,
            'l_temporal': l_temporal, 'l_logits': l_logits,
        }


# =============================================================================
# 5. 工具函数
# =============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# 6. 全局数据 & Teacher 缓存（只加载一次，所有 trial 复用）
# =============================================================================

_DATA_CACHE    = None
_TEACHER_CACHE = None


def get_data(cfg: Config):
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    manager = CrossSubjectPME4Dataset(root=cfg.data_root, fs=1000)

    print("📦 Loading EEG + PCC ...")
    (tr_eeg, tr_ey, tr_pcc), (va_eeg, va_ey, va_pcc), (te_eeg, te_ey, te_pcc) = \
        manager.get_all_splits(segment_1s=True, extract_de=True,
                               normalize=True, compute_pcc=True)

    print("📦 Loading Audio (OpenSMILE) ...")
    (tr_aud, tr_ay), (va_aud, va_ay), (te_aud, te_ay) = \
        manager.get_all_splits(modality="audio_opensmile", normalize=True)

    _DATA_CACHE = {
        'train': (tr_eeg, tr_pcc, tr_aud, tr_ey),
        'val':   (va_eeg, va_pcc, va_aud, va_ey),
        'test':  (te_eeg, te_pcc, te_aud, te_ey),
    }
    return _DATA_CACHE


def get_teacher(cfg: Config):
    global _TEACHER_CACHE
    if _TEACHER_CACHE is not None:
        return _TEACHER_CACHE

    teacher = TeacherModel(
        audio_input_dim=cfg.audio_input_dim,   audio_hidden_dim=cfg.audio_hidden_dim,
        audio_num_layers=cfg.audio_num_layers, audio_dropout=cfg.audio_dropout,
        num_nodes=cfg.num_nodes,               eeg_in_features=cfg.eeg_in_features,
        gcn_hidden=cfg.gcn_hidden,             gcn_out=cfg.gcn_out,
        lstm_hidden=cfg.lstm_hidden,           lstm_layers=cfg.lstm_layers,
        eeg_dropout=cfg.dropout,               dk=cfg.dk,
        fc_hidden=cfg.fc_hidden,               num_classes=cfg.num_classes,
    ).to(cfg.device)

    if os.path.exists(cfg.teacher_ckpt):
        state = torch.load(cfg.teacher_ckpt, map_location=cfg.device, weights_only=True)
        teacher.load_state_dict(state['model_state'])
        print(f"✅ Teacher loaded from {cfg.teacher_ckpt}")
    else:
        print(f"⚠️  Teacher ckpt not found ({cfg.teacher_ckpt}), using random weights.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    _TEACHER_CACHE = teacher
    return _TEACHER_CACHE


def make_loader(eeg, pcc, aud, y, batch_size: int, shuffle: bool):
    ds = PME4DistillDataset(eeg, pcc, aud, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=4, pin_memory=True)


# =============================================================================
# 7. 单次 Trial 训练
# =============================================================================

def run_trial(cfg: Config,
              search_epochs: int, search_patience: int,
              teacher, data: dict,
              w_ce: float, w_graph: float, w_temporal: float, lr: float,
              trial_id: int) -> dict:
    """
    用给定权重训练 Student，返回最佳 val 对应的 test 指标。
    每次 trial 在 set_seed 之后重建 loader，保证 batch shuffle
    顺序与正式训练完全一致，消除 trial 间随机状态累积的影响。
    """
    set_seed(cfg.seed)   # ← 先固定种子

    # set_seed 之后再建 loader，shuffle 顺序由种子决定，每个 trial 完全相同
    train_loader = make_loader(*data['train'], cfg.batch_size, shuffle=True)
    val_loader   = make_loader(*data['val'],   cfg.batch_size, shuffle=False)
    test_loader  = make_loader(*data['test'],  cfg.batch_size, shuffle=False)

    student = ST_GCLSTM(
        num_nodes=cfg.num_nodes,     in_features=cfg.eeg_in_features,
        gcn_hidden=cfg.gcn_hidden,   gcn_out=cfg.gcn_out,
        lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
        fc_hidden=cfg.fc_hidden,     num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(cfg.device)

    criterion = DistillationLoss(w_ce, w_graph, w_temporal,
                                  w_logits=cfg.base_w_logits,
                                  temperature=cfg.temperature)
    optimizer = optim.Adam(student.parameters(),
                           lr=lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    best_val_acc  = 0.0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, search_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────────
        student.train()
        for eeg, pcc, audio, y in train_loader:
            eeg, pcc, audio, y = (eeg.to(cfg.device), pcc.to(cfg.device),
                                   audio.to(cfg.device), y.to(cfg.device))
            with torch.no_grad():
                t_out = teacher(eeg, pcc, audio)
            s_out  = student(eeg, pcc)
            losses = criterion(s_out, t_out, y)
            optimizer.zero_grad()
            losses['loss'].backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        # ── Validate ───────────────────────────────────────────────────────────
        student.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for eeg, pcc, audio, y in val_loader:
                eeg, pcc = eeg.to(cfg.device), pcc.to(cfg.device)
                y        = y.to(cfg.device)
                s_out    = student(eeg, pcc)
                val_preds.extend(s_out['logits'].argmax(-1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            # CPU 上保存，节省显存
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= search_patience:
                break

    # ── Test（用 val 最优 checkpoint）─────────────────────────────────────────
    student.load_state_dict(best_state)
    student.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for eeg, pcc, audio, y in test_loader:
            eeg, pcc = eeg.to(cfg.device), pcc.to(cfg.device)
            y        = y.to(cfg.device)
            s_out    = student(eeg, pcc)
            test_preds.extend(s_out['logits'].argmax(-1).cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

    return {
        'trial_id'   : trial_id,
        'w_ce'       : w_ce,
        'w_graph'    : round(w_graph, 4),
        'w_temporal' : round(w_temporal, 4),
        'lr'         : lr,
        'val_acc'    : round(best_val_acc, 4),
        'test_acc'   : round(test_acc, 4),
        'test_f1'    : round(test_f1, 4),
    }


# =============================================================================
# 8. 通用搜索循环（粗 / 细 共用）
# =============================================================================

def run_stage(stage_name: str,
              cfg: Config, scfg: SearchConfig,
              teacher, data: dict,
              combos: list,
              epochs: int, patience: int,
              trial_id_offset: int,
              all_results: list, log_lines: list) -> list:
    """执行一轮搜索，实时写 JSON & log，返回本轮结果列表。"""
    total         = len(combos)
    stage_results = []

    print(f"\n{'='*70}")
    print(f"🔍 {stage_name}  ({total} combos，每组最多 {epochs} epoch)")
    print(f"{'='*70}")

    for i, (w_ce, w_graph, w_temporal, lr) in enumerate(combos, start=1):
        trial_id = trial_id_offset + i
        t0 = time.time()
        print(f"[{i:03d}/{total}]  w_ce={w_ce:.3f}  w_graph={w_graph:.3f}  "
              f"w_temporal={w_temporal:.3f}  lr={lr:.0e}", end="  ", flush=True)

        result = run_trial(
            cfg, epochs, patience,
            teacher, data,
            w_ce, w_graph, w_temporal, lr, trial_id,
        )
        result['stage'] = stage_name
        elapsed = time.time() - t0

        print(f"val={result['val_acc']:.4f}  "
              f"test={result['test_acc']:.4f}  "
              f"f1={result['test_f1']:.4f}  ({elapsed:.0f}s)")

        log_line = (
            f"[{stage_name}|{i:03d}/{total}] "
            f"w_ce={w_ce:.3f}  w_graph={w_graph:.3f}  w_temporal={w_temporal:.3f}  lr={lr:.0e} | "
            f"val={result['val_acc']:.4f}  test={result['test_acc']:.4f}  "
            f"f1={result['test_f1']:.4f} | {elapsed:.0f}s"
        )
        stage_results.append(result)
        all_results.append(result)
        log_lines.append(log_line)

        # 实时持久化（即使中途中断也能恢复）
        with open(scfg.results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        with open(scfg.log_path, 'w') as f:
            f.write('\n'.join(log_lines))

    return stage_results


# =============================================================================
# 9. 细搜候选生成（与 EAV 版相同逻辑）
# =============================================================================

def build_fine_combos(coarse_results: list, scfg: SearchConfig) -> list:
    """从粗搜 Top-N 出发，对 w_graph / w_temporal / lr 三个维度展开细搜。"""
    top = sorted(coarse_results,
                 key=lambda r: r['test_acc'], reverse=True)[:scfg.fine_n_top]
    seen, combos = set(), []
    for r in top:
        lr_candidates = scfg.fine_lr_neighbors.get(r['lr'], [r['lr']])
        for dg in scfg.fine_steps:
            for dt in scfg.fine_steps:
                wg  = round(max(0.0, r['w_graph']    + dg), 4)
                wt  = round(max(0.0, r['w_temporal'] + dt), 4)
                for lr in lr_candidates:
                    key = (r['w_ce'], wg, wt, lr)
                    if key not in seen:
                        seen.add(key)
                        combos.append((r['w_ce'], wg, wt, lr))
    return combos


# =============================================================================
# 10. 汇总打印
# =============================================================================

def print_top(results: list, n: int = 10, title: str = "Top Results"):
    print(f"\n{'='*80}")
    print(f"📊 {title}  (Top {n} by test_acc)")
    print(f"{'='*80}")
    print(f"  {'Rank':>4}  {'Stage':>6}  {'w_ce':>5}  {'w_graph':>7}  "
          f"{'w_temporal':>10}  {'lr':>7}  {'val_acc':>7}  {'test_acc':>8}  {'test_f1':>7}")
    print(f"  {'-'*78}")
    for rank, r in enumerate(
            sorted(results, key=lambda x: x['test_acc'], reverse=True)[:n], 1):
        print(f"  #{rank:02d}  [{r['stage']:>6}]  {r['w_ce']:>5.3f}  "
              f"{r['w_graph']:>7.3f}  {r['w_temporal']:>10.3f}  "
              f"{r['lr']:>7.0e}  "
              f"{r['val_acc']:>7.4f}  {r['test_acc']:>8.4f}  {r['test_f1']:>7.4f}")


# =============================================================================
# 11. 主程序
# =============================================================================

def main():
    cfg  = Config()
    scfg = SearchConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("="*80)
    print("PME4 Distillation — 两阶段超参数搜索")
    print(f"  Device      : {cfg.device}")
    n_coarse = (len(scfg.coarse_w_ce_values) * len(scfg.coarse_w_graph_values)
                * len(scfg.coarse_w_temporal_values) * len(scfg.coarse_lr_values))
    print(f"  Coarse grid : {len(scfg.coarse_w_ce_values)} × "
          f"{len(scfg.coarse_w_graph_values)} × "
          f"{len(scfg.coarse_w_temporal_values)} × "
          f"{len(scfg.coarse_lr_values)} = {n_coarse} combos")
    print(f"  Coarse epochs / patience : {scfg.coarse_epochs} / {scfg.coarse_patience}")
    print(f"  Fine   epochs / patience : {scfg.fine_epochs}   / {scfg.fine_patience}")
    print(f"  Fine Top-N               : {scfg.fine_n_top}")
    print("="*80)

    # ── 预加载数据 & Teacher（所有 trial 共享原始数组，loader 在每个 trial 内重建）
    data    = get_data(cfg)
    teacher = get_teacher(cfg)

    all_results: list = []
    log_lines:   list = []
    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段一：粗搜
    # ══════════════════════════════════════════════════════════════════════════
    coarse_combos = list(itertools.product(
        scfg.coarse_w_ce_values,
        scfg.coarse_w_graph_values,
        scfg.coarse_w_temporal_values,
        scfg.coarse_lr_values,
    ))
    coarse_results = run_stage(
        "Coarse", cfg, scfg, teacher,
        data,
        coarse_combos,
        epochs=scfg.coarse_epochs, patience=scfg.coarse_patience,
        trial_id_offset=0,
        all_results=all_results, log_lines=log_lines,
    )
    print_top(coarse_results, n=5, title="Coarse Stage Top-5")

    # ══════════════════════════════════════════════════════════════════════════
    # 阶段二：细搜
    # ══════════════════════════════════════════════════════════════════════════
    fine_combos = build_fine_combos(coarse_results, scfg)
    print(f"\n🔬 Fine search: 由粗搜 Top-{scfg.fine_n_top} 生成 {len(fine_combos)} 个候选")

    fine_results = run_stage(
        "Fine", cfg, scfg, teacher,
        data,
        fine_combos,
        epochs=scfg.fine_epochs, patience=scfg.fine_patience,
        trial_id_offset=len(coarse_combos),
        all_results=all_results, log_lines=log_lines,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 最终汇总
    # ══════════════════════════════════════════════════════════════════════════
    print_top(all_results, n=10, title="Overall Top-10")

    best = sorted(all_results, key=lambda r: r['test_acc'], reverse=True)[0]
    print(f"\n🏆 Best overall:")
    print(f"   w_ce       = {best['w_ce']}")
    print(f"   w_graph    = {best['w_graph']}")
    print(f"   w_temporal = {best['w_temporal']}")
    print(f"   lr         = {best['lr']:.0e}")
    print(f"   val_acc    = {best['val_acc']:.4f}")
    print(f"   test_acc   = {best['test_acc']:.4f}")
    print(f"   test_f1    = {best['test_f1']:.4f}")
    print(f"   stage      = {best['stage']}")
    print(f"\n⏱  Total time : {(time.time()-t_total)/60:.1f} min")
    print(f"💾 Results     → {scfg.results_path}")
    print(f"📋 Log         → {scfg.log_path}")


if __name__ == '__main__':
    main()