import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

# 加入路徑以便加載 dataset
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')
from KDbaseline.model.AMBOKD_Model import AMBOKDModel
from dataset.dataset import CrossSubjectMultiModalDataset


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
# Dataset
# =============================================================================

class AlignedDistillDataset(Dataset):
    def __init__(self, eeg, pcc, labels, audio, vision):
        self.eeg    = eeg
        self.pcc    = pcc
        self.labels = labels
        self.audio  = audio
        self.vision = vision

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.eeg[idx], self.pcc[idx],
                self.audio[idx], self.vision[idx],
                self.labels[idx])


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


# =============================================================================
# 主程序
# =============================================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)
    
    # ── 数据加载 ─────────────────────────────────────────────────────────────
    print("\n📦 Loading data ...")
    manager = CrossSubjectMultiModalDataset(
        cfg.data_root_dict,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )

    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)

    (tr_aud, _), (va_aud, _), (te_aud, _) = \
        manager.get_all_splits('audio', normalize=True)

    (tr_vis, _), (va_vis, _), (te_vis, _) = \
        manager.get_all_splits('vision', normalize=True)

    def make_loader(eeg, pcc, y, aud, vis, shuffle):
        ds = AlignedDistillDataset(eeg, pcc, y, aud, vis)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = make_loader(tr_eeg, tr_pcc, tr_y, tr_aud, tr_vis, True)
    val_loader   = make_loader(va_eeg, va_pcc, va_y, va_aud, va_vis, False)
    test_loader  = make_loader(te_eeg, te_pcc, te_y, te_aud, te_vis, False)
    
    # ── 模型 ──────────────────────────────────────────────────────────────────
    print("🤖 Initializing Model...")
    model = AMBOKDModel(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    
    best_val_acc_eeg = 0.0
    patience_count = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f"\n🚀 AMBOKD Training  seed={cfg.seed}")
    print(f"{'='*90}")
    print(f"  {'Ep':>4}  {'LR':>8}  "
          f"{'Tr-Loss':>8} {'Tr-Acc':>7} {'Tr-F1':>7} {'Tr-AccE':>7}  "
          f"{'Va-Loss':>8} {'Va-Acc':>7} {'Va-F1':>7} {'Va-AccE':>7}")
    print(f"{'='*90}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        tr = run_phase(cfg, model, train_loader, optimizer, epoch)
        va = run_phase(cfg, model, val_loader, epoch=epoch)

        scheduler.step(va['acc_eeg'])
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        flag = ''
        if va['acc_eeg'] > best_val_acc_eeg:
            best_val_acc_eeg = va['acc_eeg']
            patience_count = 0
            torch.save({'epoch'      : epoch,
                        'model_state': model.state_dict(),
                        'val_acc_eeg': best_val_acc_eeg},
                       cfg.model_ckpt)
            flag = '  ✅'
        else:
            patience_count += 1
            flag = f'  ({patience_count}/{cfg.patience})'

        print(f"  {epoch:4d}  {lr_now:8.2e}  "
              f"{tr['loss']:8.4f} {tr['acc']:7.4f} {tr['f1']:7.4f} {tr['acc_eeg']:7.4f}  "
              f"{va['loss']:8.4f} {va['acc']:7.4f} {va['f1']:7.4f} {va['acc_eeg']:7.4f}"
              f"{flag}  [{elapsed:.1f}s]")

        if patience_count >= cfg.patience:
            print(f"\n⏹️  Early stopping at epoch {epoch}")
            break

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print(f"\n🔍 Best val acc (EEG): {best_val_acc_eeg:.4f}  →  {cfg.model_ckpt}")
    model.load_state_dict(
        torch.load(cfg.model_ckpt, map_location=device)['model_state'])
    model.eval()

    te = run_phase(cfg, model, test_loader, epoch=0)

    print(f"\n{'='*50}")
    print(f"  Test Results:")
    print(f"  ── Fused Branch ──")
    print(f"    Acc  : {te['acc']:.4f}")
    print(f"    F1   : {te['f1']:.4f}")
    print(f"  ── EEG Branch ──")
    print(f"    Acc  : {te['acc_eeg']:.4f}")
    print(f"    F1   : {te['f1_eeg']:.4f}")
    print(f"  ── Loss ──")
    print(f"    Total: {te['loss']:.4f}")
    print(f"    Fused: {te['ce_fused']:.4f}")
    print(f"    EEG  : {te['ce_eeg']:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
