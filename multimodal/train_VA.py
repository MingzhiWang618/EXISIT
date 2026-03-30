import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')
from dataset.dataset import CrossSubjectMultiModalDataset, EEGDataset
from multimodal.model.VA import ImprovedAudioVisualBiLSTM


# =============================================================================
# 双模态 DataLoader 构建
# =============================================================================

def create_audio_visual_dataloaders(
        audio_root:          str,
        vision_root:         str,
        batch_size:          int  = 64,
        normalize:           bool = True,
        audio_feature_type:  str  = 'opensmile',
        vision_feature_type: str  = 'openface',
        vision_max_len:      int  = None,
        num_workers:         int  = 4,
):
    """
    分别加载 audio / vision，验证标签对齐后合并为双模态 DataLoader。
    每批返回: (audio [B, T_a, F_a], vision [B, T_v, F_v], y [B])
    """
    data_root_dict = {'audio': audio_root, 'vision': vision_root}
    ds = CrossSubjectMultiModalDataset(
        data_root_dict,
        audio_feature_type  = audio_feature_type,
        vision_feature_type = vision_feature_type,
    )

    loaders = []
    for split, shuffle in [('train', True), ('val', False), ('test', False)]:
        aud_x, aud_y = ds.load_data_by_split(split, 'audio')
        vis_x, vis_y = ds.load_data_by_split(split, 'vision')

        assert len(aud_y) == len(vis_y), \
            f"[{split}] audio/vision 样本数不一致: {len(aud_y)} vs {len(vis_y)}"
        assert (np.array(aud_y) == np.array(vis_y)).all(), \
            f"[{split}] audio/vision 标签不对齐！"

        loaders.append((aud_x, vis_x, aud_y))

    # 归一化（audio / vision 分别用各自训练集统计量）
    if normalize:
        train_aud, val_aud, test_aud = ds.normalize_by_modality(
            loaders[0][0], loaders[1][0], loaders[2][0], 'audio')
        train_vis, val_vis, test_vis = ds.normalize_by_modality(
            loaders[0][1], loaders[1][1], loaders[2][1], 'vision')
        loaders = [
            (train_aud, train_vis, loaders[0][2]),
            (val_aud,   val_vis,   loaders[1][2]),
            (test_aud,  test_vis,  loaders[2][2]),
        ]

    # vision 变长 list → pad 为 ndarray，确定 max_len（用训练集）
    if vision_max_len is None:
        vision_max_len = max(s.shape[0] for s in loaders[0][1])
        print(f"📐 vision max_len (from train): {vision_max_len}")

    def pad_vision(vis_list, max_len):
        F   = vis_list[0].shape[-1]
        arr = np.zeros((len(vis_list), max_len, F), dtype=np.float32)
        for i, s in enumerate(vis_list):
            t = min(s.shape[0], max_len)
            arr[i, :t] = s[:t]
        return arr

    result_loaders = []
    for idx, (aud_x, vis_x, y) in enumerate(loaders):
        shuffle = (idx == 0)

        # vision list → ndarray
        if isinstance(vis_x, list):
            vis_x = pad_vision(vis_x, vision_max_len)

        aud_t = torch.tensor(aud_x, dtype=torch.float32)
        vis_t = torch.tensor(vis_x, dtype=torch.float32)
        y_t   = torch.tensor(np.array(y), dtype=torch.long)

        dataset = TensorDataset(aud_t, vis_t, y_t)
        result_loaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True)
        )

    return result_loaders[0], result_loaders[1], result_loaders[2]


# =============================================================================
# 训练 / 评估
# =============================================================================

import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for audio, vision, y in loader:
        audio, vision, y = audio.to(device), vision.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(audio, vision)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for audio, vision, y in loader:
        audio, vision, y = audio.to(device), vision.to(device), y.to(device)
        logits = model(audio, vision)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / total, correct / total


# =============================================================================
# 主程序
# =============================================================================

def main():
    # ── 配置 ──────────────────────────────────────────────────────────────────
    AUDIO_ROOT  = '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw'
    VISION_ROOT = '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace'

    AUDIO_DIM   = 25    # OpenSMILE eGeMAPS LLD 维度，按实际填写
    VISION_DIM  = 161   # OpenFace gaze+landmark+AU 维度，按实际填写
    NUM_CLASSES = 5
    BATCH_SIZE  = 64
    EPOCHS      = 100
    LR          = 5e-5
    WEIGHT_DECAY= 1e-4
    SAVE_PATH   = 'best_av_bilstm.pth'
    PATIENCE    = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_audio_visual_dataloaders(
        audio_root          = AUDIO_ROOT,
        vision_root         = VISION_ROOT,
        batch_size          = BATCH_SIZE,
        normalize           = True,
        audio_feature_type  = 'opensmile',
        vision_feature_type = 'openface',
    )

    # 从第一个 batch 自动获取实际特征维度
    _a, _v, _ = next(iter(train_loader))
    AUDIO_DIM  = _a.shape[-1]
    VISION_DIM = _v.shape[-1]
    print(f"📊 audio dim: {AUDIO_DIM}, vision dim: {VISION_DIM}")

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = ImprovedAudioVisualBiLSTM(
        audio_input_dim=AUDIO_DIM,
        vision_input_dim=VISION_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=64,
        num_layers=2,
        dropout=0.5,
        rnn_type='lstm',
        pooling='attention',          # 使用注意力池化
        fusion='concat',                
        fusion_output_dim=128,
        classifier_hidden_dims=[64, 32],
        use_batchnorm=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── 训练 ──────────────────────────────────────────────────────────────────
    best_val_acc           = 0.0
    best_epoch             = 0
    epochs_no_improve      = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        improved = va_acc > best_val_acc
        if improved:
            best_val_acc      = va_acc
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"Epoch {epoch:03d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}"
              + (" ✅" if improved else f" ({epochs_no_improve}/{PATIENCE})"))

        epochs_no_improve += (0 if improved else 1)
        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 早停触发，最佳 epoch: {best_epoch}")
            break

    print(f"\n🏆 Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")

    # ── 测试 ──────────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(SAVE_PATH))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"📊 Test | loss {te_loss:.4f} acc {te_acc:.4f}")


if __name__ == '__main__':
    main()