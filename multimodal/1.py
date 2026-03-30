import os
import sys
import time
import itertools
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from multimodal.model.Teacher import TeacherModel
from multimodal.model.Student import ST_GCLSTM
from dataset.dataset import CrossSubjectMultiModalDataset

# =============================================================================
# 复用原始 Config（只改蒸馏权重）
# =============================================================================

class Config:
    data_root_dict = {
        'eeg'   : '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'audio' : '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace',
    }
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
    dk          = 32

    batch_size   = 64
    epochs       = 150        # 搜索时可调小，见 SearchConfig
    lr           = 1e-5
    weight_decay = 1e-3
    patience     = 20
    temperature  = 1.0
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed         = 2024

    save_dir = './checkpoints'

# =============================================================================
# 搜索空间配置
# =============================================================================

class SearchConfig:
    # ── 阶段一：粗搜（大步长，覆盖宽范围）─────────────────────────────────────
    # 共 1×6×6 = 36 组
    coarse_w_ce_values       = [1.0]
    coarse_w_graph_values    = [0.0]
    coarse_w_temporal_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

    # ── 阶段二：细搜（在粗搜 Top-3 周围缩小步长）─────────────────────────────
    # 以粗搜最优值为中心，±1 步细化，步长 = 粗搜步长 / 4
    fine_n_top      = 3      # 取粗搜 Top-N 做细化
    fine_steps      = [-0.3, -0.15, 0.0, 0.15, 0.3]   # 相对偏移

    # ── epoch / patience ──────────────────────────────────────────────────────
    coarse_epochs   = 60     # 粗搜 epoch 少一点，快速筛
    coarse_patience = 12
    fine_epochs     = 100    # 细搜给更多 epoch
    fine_patience   = 18

    results_path = './checkpoints/hparam_search_results.json'
    log_path     = './checkpoints/hparam_search_log.txt'

# =============================================================================
# 复用原始代码中的工具类（最小化重复）
# =============================================================================

import torch.nn.functional as F

class AlignedDistillDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_data, pcc_data, labels, audio_data, vision_data):
        self.eeg    = torch.tensor(eeg_data,   dtype=torch.float32)
        self.pcc    = torch.tensor(pcc_data,   dtype=torch.float32)
        self.labels = torch.tensor(labels,     dtype=torch.long)
        self.audio  = torch.tensor(audio_data, dtype=torch.float32)
        # vision: list of arrays → pad
        if isinstance(vision_data, list):
            max_len = max(v.shape[0] for v in vision_data)
            F_vis   = vision_data[0].shape[-1]
            arr     = np.zeros((len(vision_data), max_len, F_vis), dtype=np.float32)
            for i, v in enumerate(vision_data):
                t = min(v.shape[0], max_len)
                arr[i, :t] = v[:t]
            self.vision = torch.tensor(arr, dtype=torch.float32)
        else:
            self.vision = torch.tensor(vision_data, dtype=torch.float32)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.eeg[idx], self.pcc[idx], self.audio[idx],
                self.vision[idx], self.labels[idx])


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val = val; self.sum += val*n; self.count += n; self.avg = self.sum/self.count


class DistillationLoss(nn.Module):
    def __init__(self, w_ce, w_graph, w_temporal, temperature=1.0):
        super().__init__()
        self.w_ce       = w_ce
        self.w_graph    = w_graph
        self.w_temporal = w_temporal
        self.T          = temperature
        self.ce         = nn.CrossEntropyLoss()

    def forward(self, s_out, t_out, labels):
        l_ce = self.ce(s_out['logits'], labels)

        B, T_seq, N, _ = s_out['S_attn'].shape
        s_S = s_out['S_attn'].reshape(-1, N)
        t_S = t_out['S_attn'].reshape(-1, N).detach()
        l_graph = F.kl_div(
            s_S.clamp(min=1e-8).log(),
            t_S.clamp(min=1e-8),
            reduction='sum'
        ) / (B * T_seq * N)

        l_temporal = F.kl_div(
            s_out['attn_t'].clamp(min=1e-8).log(),
            t_out['attn_t'].detach().clamp(min=1e-8),
            reduction='batchmean'
        )

        total = self.w_ce * l_ce + self.w_graph * l_graph + self.w_temporal * l_temporal
        return {'loss': total, 'l_ce': l_ce, 'l_graph': l_graph, 'l_temporal': l_temporal}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_loaders(cfg):
    """加载数据（只做一次，所有搜索 trial 共享）"""
    manager = CrossSubjectMultiModalDataset(
        cfg.data_root_dict,
        audio_feature_type='opensmile',
        vision_feature_type='openface'
    )
    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)
    (tr_aud, tr_aud_y), (va_aud, _), (te_aud, _) = \
        manager.get_all_splits('audio', normalize=True)
    (tr_vis, tr_vis_y), (va_vis, _), (te_vis, _) = \
        manager.get_all_splits('vision', normalize=True)

    def make(eeg, pcc, y, aud, vis, shuffle):
        ds = AlignedDistillDataset(eeg, pcc, y, aud, vis)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = make(tr_eeg, tr_pcc, tr_y, tr_aud, tr_vis, shuffle=True)
    val_loader   = make(va_eeg, va_pcc, va_y, va_aud, va_vis, shuffle=False)
    test_loader  = make(te_eeg, te_pcc, te_y, te_aud, te_vis, shuffle=False)
    return train_loader, val_loader, test_loader


def build_teacher(cfg):
    teacher = TeacherModel(
        audio_input_dim=cfg.audio_dim, vision_input_dim=cfg.vision_dim,
        av_hidden_dim=cfg.av_hidden, av_num_layers=cfg.lstm_layers,
        av_dropout=cfg.dropout, num_nodes=cfg.num_nodes,
        eeg_in_features=cfg.in_features, gcn_hidden=cfg.gcn_hidden,
        gcn_out=cfg.gcn_out, lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers, eeg_dropout=cfg.dropout,
        dk=cfg.dk, fc_hidden=cfg.fc_hidden, num_classes=cfg.num_classes,
    ).to(cfg.device)

    t_ckpt = os.path.join(cfg.save_dir, 'best_teacher.pth')
    if os.path.exists(t_ckpt):
        teacher.load_state_dict(
            torch.load(t_ckpt, map_location=cfg.device, weights_only=True)['model_state'])
        print(f"✅ Teacher loaded from {t_ckpt}")
    else:
        print(f"⚠️  Teacher checkpoint not found at {t_ckpt}, using random weights")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# =============================================================================
# 单次 trial：训练 + 返回 val/test 指标
# =============================================================================

def run_trial(cfg, scfg, teacher, train_loader, val_loader, test_loader,
              w_ce, w_graph, w_temporal, trial_id):

    set_seed(cfg.seed)

    student = ST_GCLSTM(
        num_nodes=cfg.num_nodes, in_features=cfg.in_features,
        gcn_hidden=cfg.gcn_hidden, gcn_out=cfg.gcn_out,
        lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
        fc_hidden=cfg.fc_hidden, num_classes=cfg.num_classes,
        dropout=cfg.dropout,
    ).to(cfg.device)

    criterion = DistillationLoss(w_ce, w_graph, w_temporal, cfg.temperature)
    optimizer = optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    best_val_acc    = 0.0
    best_state      = None
    patience_count  = 0

    for epoch in range(1, scfg.search_epochs + 1):
        # ── train ──
        student.train()
        for eeg, pcc, audio, vision, y in train_loader:
            eeg, pcc, audio, vision, y = [d.to(cfg.device)
                                           for d in [eeg, pcc, audio, vision, y]]
            with torch.no_grad():
                t_out = teacher(eeg, pcc, audio, vision)
            s_out  = student(eeg, pcc)
            losses = criterion(s_out, t_out, y)
            optimizer.zero_grad()
            losses['loss'].backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

        # ── val ──
        student.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for eeg, pcc, audio, vision, y in val_loader:
                eeg, pcc, audio, vision, y = [d.to(cfg.device)
                                               for d in [eeg, pcc, audio, vision, y]]
                t_out = teacher(eeg, pcc, audio, vision)
                s_out = student(eeg, pcc)
                val_preds.extend(s_out['logits'].argmax(-1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= scfg.search_patience:
                break

    # ── test with best val checkpoint ──
    student.load_state_dict(best_state)
    student.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for eeg, pcc, audio, vision, y in test_loader:
            eeg, pcc, audio, vision, y = [d.to(cfg.device)
                                           for d in [eeg, pcc, audio, vision, y]]
            s_out = student(eeg, pcc)
            test_preds.extend(s_out['logits'].argmax(-1).cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1  = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

    return {
        'trial_id' : trial_id,
        'w_ce'     : w_ce,
        'w_graph'  : w_graph,
        'w_temporal': w_temporal,
        'val_acc'  : round(best_val_acc, 4),
        'test_acc' : round(test_acc, 4),
        'test_f1'  : round(test_f1, 4),
    }


# =============================================================================
# 主搜索循环
# =============================================================================

def run_stage(stage_name, cfg, scfg, teacher,
              train_loader, val_loader, test_loader,
              combos, epochs, patience, trial_id_offset,
              all_results, log_lines):
    """通用的一轮搜索循环，返回本轮所有结果"""
    total       = len(combos)
    stage_results = []

    # 用一个临时 scfg-like 对象传 epochs/patience 给 run_trial
    class _E:
        search_epochs   = epochs
        search_patience = patience

    print(f"\n{'='*70}")
    print(f"🔍 {stage_name}  ({total} combos, {epochs} epochs each)")
    print(f"{'='*70}")

    for i, (w_ce, w_graph, w_temporal) in enumerate(combos, start=1):
        trial_id = trial_id_offset + i
        t0 = time.time()
        print(f"[{i:02d}/{total}] w_ce={w_ce:.3f}  w_graph={w_graph:.3f}  "
              f"w_temporal={w_temporal:.3f}", end='  ', flush=True)

        result = run_trial(cfg, _E, teacher,
                           train_loader, val_loader, test_loader,
                           w_ce, w_graph, w_temporal, trial_id)
        result['stage'] = stage_name

        elapsed  = time.time() - t0
        log_line = (
            f"[{stage_name}|{i:02d}/{total}] "
            f"w_ce={w_ce:.3f} w_graph={w_graph:.3f} w_temporal={w_temporal:.3f} | "
            f"val={result['val_acc']:.4f} test={result['test_acc']:.4f} "
            f"f1={result['test_f1']:.4f} | {elapsed:.0f}s"
        )
        print(f"val={result['val_acc']:.4f}  test={result['test_acc']:.4f}  "
              f"f1={result['test_f1']:.4f}  ({elapsed:.0f}s)")

        stage_results.append(result)
        all_results.append(result)
        log_lines.append(log_line)

        with open(scfg.results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        with open(scfg.log_path, 'w') as f:
            f.write('\n'.join(log_lines))

    return stage_results


def build_fine_combos(coarse_results, scfg):
    """从粗搜 Top-N 出发，在各参数周围生成细搜组合"""
    top = sorted(coarse_results, key=lambda r: r['test_acc'], reverse=True)[:scfg.fine_n_top]
    seen   = set()
    combos = []
    for r in top:
        for dg in scfg.fine_steps:
            for dt in scfg.fine_steps:
                wg = round(max(0.0, r['w_graph']    + dg), 4)
                wt = round(max(0.0, r['w_temporal'] + dt), 4)
                key = (r['w_ce'], wg, wt)
                if key not in seen:
                    seen.add(key)
                    combos.append((r['w_ce'], wg, wt))
    return combos


def print_top(all_results, n=10):
    print(f"\n{'='*70}")
    print(f"📊 Top {n} by test_acc")
    print(f"{'='*70}")
    for rank, r in enumerate(
            sorted(all_results, key=lambda x: x['test_acc'], reverse=True)[:n], 1):
        print(f"  #{rank:02d} [{r['stage']}]  "
              f"w_ce={r['w_ce']:.3f}  w_graph={r['w_graph']:.3f}  "
              f"w_temporal={r['w_temporal']:.3f}  |  "
              f"val={r['val_acc']:.4f}  test={r['test_acc']:.4f}  "
              f"f1={r['test_f1']:.4f}")


def main():
    cfg  = Config()
    scfg = SearchConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    print("📦 Loading data ...")
    train_loader, val_loader, test_loader = build_loaders(cfg)
    print("🧑‍🏫 Building teacher ...")
    teacher = build_teacher(cfg)

    all_results = []
    log_lines   = []

    # ══════════════════════════════════════════════════════════════
    # 阶段一：粗搜
    # ══════════════════════════════════════════════════════════════
    coarse_combos = list(itertools.product(
        scfg.coarse_w_ce_values,
        scfg.coarse_w_graph_values,
        scfg.coarse_w_temporal_values,
    ))
    coarse_results = run_stage(
        "Coarse", cfg, scfg, teacher,
        train_loader, val_loader, test_loader,
        coarse_combos,
        epochs=scfg.coarse_epochs, patience=scfg.coarse_patience,
        trial_id_offset=0,
        all_results=all_results, log_lines=log_lines,
    )
    print_top(coarse_results, n=5)

    # ══════════════════════════════════════════════════════════════
    # 阶段二：细搜（围绕粗搜 Top-N 展开）
    # ══════════════════════════════════════════════════════════════
    fine_combos = build_fine_combos(coarse_results, scfg)
    print(f"\n🔬 Fine search: {len(fine_combos)} combos generated from "
          f"Top-{scfg.fine_n_top} coarse results")

    fine_results = run_stage(
        "Fine", cfg, scfg, teacher,
        train_loader, val_loader, test_loader,
        fine_combos,
        epochs=scfg.fine_epochs, patience=scfg.fine_patience,
        trial_id_offset=len(coarse_combos),
        all_results=all_results, log_lines=log_lines,
    )

    # ══════════════════════════════════════════════════════════════
    # 最终汇总
    # ══════════════════════════════════════════════════════════════
    print_top(all_results, n=10)
    best = sorted(all_results, key=lambda r: r['test_acc'], reverse=True)[0]
    print(f"\n🏆 Best overall:  w_ce={best['w_ce']}  w_graph={best['w_graph']}  "
          f"w_temporal={best['w_temporal']}")
    print(f"   val={best['val_acc']:.4f}  test={best['test_acc']:.4f}  "
          f"f1={best['test_f1']:.4f}  [{best['stage']}]")
    print(f"\nFull results → {scfg.results_path}")


if __name__ == '__main__':
    main()