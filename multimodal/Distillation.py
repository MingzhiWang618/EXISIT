import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# 确保路径正确
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

from multimodal.model.Teacher import TeacherModel
from multimodal.model.Student import ST_GCLSTM
from dataset.dataset import create_single_modality_dataloaders, CrossSubjectMultiModalDataset

# =============================================================================
# 1. 严格配置 (与 train_student.py 保持物理一致)
# =============================================================================

class Config:
    # ── 数据 ──────────────────────────────────────────────────────────────────
    data_root_dict = {
        'eeg'   : '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'audio' : '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace',
    }
    # 特别注意：为了对齐，我们必须使用与 student 脚本相同的原始 Loader 逻辑
    modality    = 'eeg'
    extract_de  = True
    segment_1s  = True
    normalize   = True
    compute_pcc = True
    fs          = 100

    # ── 模型参数 ──────────────────────────────────────────────────────────────
    num_nodes   = 30
    in_features = 5
    gcn_hidden  = 64
    gcn_out     = 64
    lstm_hidden = 64
    lstm_layers = 1
    fc_hidden   = 64
    num_classes = 5
    dropout     = 0.5
    
    # Teacher 专用
    audio_dim   = 25
    vision_dim  = 161
    av_hidden   = 64
    dk          = 32

    # ── 训练参数 ──────────────────────────────────────────────────────────────
    batch_size   = 64
    epochs       = 150
    lr           = 5e-4
    weight_decay = 1e-3
    patience     = 20
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed         = 2024


    # ── 蒸馏权重 (设为 0 则理论结果应与 train_student 完全一致) ─────────────────
    # 基础权重
    base_w_ce        = 1.0
    base_w_graph     = 1.0
    base_w_temporal  = 0.0
    base_w_logits    = 0.0
    temperature      = 1.0
    
    # 前期增强权重
    initial_w_graph  = 0.0  # 前期graph权重
    initial_w_temporal = 0  # 前期temporal权重
    weight_decay_epoch = 0  # 权重衰减的epoch数

    save_dir  = './checkpoints'
    exp_name  = 'st_gclstm_distill'

# =============================================================================
# 2. 核心蒸馏损失逻辑
# =============================================================================


def spectral_loss(s_R, t_R):
    B, T, N, _ = s_R.shape
    s = s_R.reshape(B * T, N, N)
    t = t_R.reshape(B * T, N, N).detach()
    
    s_sv = torch.linalg.svdvals(s)   # [B*T, N] 全部奇异值
    t_sv = torch.linalg.svdvals(t)
    
    # 按奇异值大小加权，大的奇异值权重高
    weights = t_sv / (t_sv.sum(dim=-1, keepdim=True) + 1e-8)
    
    return (weights * (s_sv - t_sv).pow(2)).sum(dim=-1).mean()


class DistillationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.base_w_ce = cfg.base_w_ce
        self.base_w_graph = cfg.base_w_graph
        self.base_w_temporal = cfg.base_w_temporal
        self.base_w_logits = cfg.base_w_logits
        self.initial_w_graph = cfg.initial_w_graph
        self.initial_w_temporal = cfg.initial_w_temporal
        self.weight_decay_epoch = cfg.weight_decay_epoch
        self.T = cfg.temperature
        self.ce = nn.CrossEntropyLoss()

    def get_weights(self, epoch):
        """根据当前epoch动态调整权重"""
        # 计算权重衰减因子
        if epoch < self.weight_decay_epoch:
            # 线性衰减
            decay_factor = epoch / self.weight_decay_epoch
            w_graph = self.initial_w_graph * (1 - decay_factor) + self.base_w_graph * decay_factor
            w_temporal = self.initial_w_temporal * (1 - decay_factor) + self.base_w_temporal * decay_factor
        else:
            # 达到基础权重
            w_graph = self.base_w_graph
            w_temporal = self.base_w_temporal
        
        return {
            'w_ce': self.base_w_ce,
            'w_graph': w_graph,
            'w_temporal': w_temporal,
            'w_logits': self.base_w_logits
        }

    def forward(self, s_out, t_out, labels, epoch=0):
        weights = self.get_weights(epoch)

        # Hard label loss
        l_ce = self.ce(s_out['logits'], labels)

        # Graph KL：用 S_attn 替代 R，每行是 N=30 维概率分布
        B, T_seq, N, _ = s_out['S_attn'].shape
        s_S = s_out['S_attn'].reshape(-1, N)                          # [B*T*N, N]
        t_S = t_out['S_attn'].reshape(-1, N).detach()                 # [B*T*N, N]
        l_graph = F.kl_div(
            s_S.clamp(min=1e-8).log(),
            t_S.clamp(min=1e-8),
            reduction='sum'
        ) / (B * T_seq * N)

        # Temporal KL
        l_temporal = F.kl_div(
            s_out['attn_t'].clamp(min=1e-8).log(),
            t_out['attn_t'].detach().clamp(min=1e-8),
            reduction='batchmean'
        )

        # Logits KL
        t_soft     = F.softmax(t_out['eeg_logits'] / self.T, dim=-1).detach()
        s_log_soft = F.log_softmax(s_out['logits'] / self.T, dim=-1)
        l_logits   = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (self.T ** 2)

        total = (weights['w_ce']       * l_ce
            + weights['w_graph']    * l_graph
            + weights['w_temporal'] * l_temporal
            + weights['w_logits']   * l_logits)

        return {
            'loss'      : total,
            'l_ce'      : l_ce,
            'l_graph'   : l_graph,
            'l_temporal': l_temporal,
            'l_logits'  : l_logits,
            'weights'   : weights
        }

# =============================================================================
# 3. 对齐训练工具
# =============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0.0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

# =============================================================================
# 4. 统一数据封装 (确保顺序对齐的关键)
# =============================================================================

class AlignedDistillDataset(torch.utils.data.Dataset):
    """
    通过索引同步 EEG, Audio, Vision，确保样本顺序不因多模态加载而被打乱
    """
    def __init__(self, eeg_data, pcc_data, labels, audio_data, vision_data):
        self.eeg = eeg_data
        self.pcc = pcc_data
        self.labels = labels
        self.audio = audio_data
        self.vision = vision_data

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (self.eeg[idx], self.pcc[idx], self.audio[idx], 
                self.vision[idx], self.labels[idx])

# =============================================================================
# 5. 训练与评估函数
# =============================================================================

def run_epoch(cfg, teacher, student, loader, criterion, optimizer=None, epoch=0):
    is_train = optimizer is not None
    student.train() if is_train else student.eval()
    teacher.eval()
    
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    graph_meter = AverageMeter()
    temporal_meter = AverageMeter()
    logits_meter = AverageMeter()
    all_preds, all_labels = [], []

    for eeg, pcc, audio, vision, y in loader:
        eeg, pcc, audio, vision, y = [d.to(cfg.device) for d in [eeg, pcc, audio, vision, y]]
        
        with torch.no_grad():
            t_out = teacher(eeg, pcc, audio, vision)
        
        if is_train:
            s_out = student(eeg, pcc)
        else:
            with torch.no_grad():
                s_out = student(eeg, pcc)
        losses = criterion(s_out, t_out, y, epoch)
        loss = losses['loss']

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

        loss_meter.update(loss.item(), eeg.size(0))
        ce_meter.update(losses['l_ce'].item(), eeg.size(0))
        graph_meter.update(losses['l_graph'].item(), eeg.size(0))
        temporal_meter.update(losses['l_temporal'].item(), eeg.size(0))
        logits_meter.update(losses['l_logits'].item(), eeg.size(0))
        all_preds.extend(s_out['logits'].argmax(dim=-1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {
        'loss': loss_meter.avg, 
        'acc': acc, 
        'f1': f1,
        'l_ce': ce_meter.avg,
        'l_graph': graph_meter.avg,
        'l_temporal': temporal_meter.avg,
        'l_logits': logits_meter.avg
    }

# =============================================================================
# 6. 主程序
# =============================================================================

def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1. 数据准备 (保持之前的对齐逻辑)
    manager = CrossSubjectMultiModalDataset(
        cfg.data_root_dict,
        audio_feature_type='opensmile',
        vision_feature_type='openface'
    )
    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)
    (tr_aud, tr_aud_y), (va_aud, va_aud_y), (te_aud, _) = manager.get_all_splits('audio', normalize=True)
    (tr_vis, tr_vis_y), (va_vis, va_vis_y), (te_vis, _) = manager.get_all_splits('vision', normalize=True)



    train_loader = DataLoader(AlignedDistillDataset(tr_eeg, tr_pcc, tr_y, tr_aud, tr_vis), 
                              batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(AlignedDistillDataset(va_eeg, va_pcc, va_y, va_aud, va_vis), 
                              batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(AlignedDistillDataset(te_eeg, te_pcc, te_y, te_aud, te_vis), 
                              batch_size=cfg.batch_size, shuffle=False)

    # 2. 模型初始化
    student = ST_GCLSTM(
        num_nodes=cfg.num_nodes, in_features=cfg.in_features, gcn_hidden=cfg.gcn_hidden,
        gcn_out=cfg.gcn_out, lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers,
        fc_hidden=cfg.fc_hidden, num_classes=cfg.num_classes, dropout=cfg.dropout
    ).to(cfg.device)

    teacher = TeacherModel(
        audio_input_dim=cfg.audio_dim, vision_input_dim=cfg.vision_dim, av_hidden_dim=cfg.av_hidden,
        av_num_layers=cfg.lstm_layers, av_dropout=cfg.dropout, num_nodes=cfg.num_nodes,
        eeg_in_features=cfg.in_features, gcn_hidden=cfg.gcn_hidden, gcn_out=cfg.gcn_out,
        lstm_hidden=cfg.lstm_hidden, lstm_layers=cfg.lstm_layers, eeg_dropout=cfg.dropout,
        dk=cfg.dk, fc_hidden=cfg.fc_hidden, num_classes=cfg.num_classes
    ).to(cfg.device)

    t_ckpt = os.path.join(cfg.save_dir, 'best_teacher.pth')
    if os.path.exists(t_ckpt):
        teacher.load_state_dict(torch.load(t_ckpt, map_location=cfg.device, weights_only=True)['model_state'])
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # 3. 优化与训练
    criterion = DistillationLoss(cfg)
    optimizer = optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    best_val_acc = 0.0
    patience_count = 0
    ckpt_path = os.path.join(cfg.save_dir, 'best_student_distill.pt')

    print(f"\n🚀 Start Distillation Training. W_CE={cfg.base_w_ce}, W_Graph={cfg.base_w_graph}, W_Temporal={cfg.base_w_temporal}, Seed={cfg.seed}")
    print("-" * 90)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_res = run_epoch(cfg, teacher, student, train_loader, criterion, optimizer, epoch)
        val_res   = run_epoch(cfg, teacher, student, val_loader, criterion, epoch=epoch)
        
        scheduler.step(val_res['acc'])
        elapsed = time.time() - t0
        
        # 获取当前权重
        current_weights = criterion.get_weights(epoch)
        
        print(f"Epoch {epoch:03d} | Train Acc: {train_res['acc']:.4f} | Val Acc: {val_res['acc']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")
        print(f"        Weights: CE={current_weights['w_ce']:.2f}, Graph={current_weights['w_graph']:.2f}, Temporal={current_weights['w_temporal']:.2f}, Logits={current_weights['w_logits']:.2f}")
        print(f"        Train Loss: {train_res['loss']:.4f} (CE: {train_res['l_ce']:.4f}, Graph: {train_res['l_graph']:.4f}, Temporal: {train_res['l_temporal']:.4f}, Logits: {train_res['l_logits']:.4f})")
        print(f"        Val Loss:   {val_res['loss']:.4f} (CE: {val_res['l_ce']:.4f}, Graph: {val_res['l_graph']:.4f}, Temporal: {val_res['l_temporal']:.4f}, Logits: {val_res['l_logits']:.4f})")

        if val_res['acc'] > best_val_acc:
            best_val_acc = val_res['acc']
            patience_count = 0
            torch.save({'model_state': student.state_dict(), 'val_acc': best_val_acc}, ckpt_path)
            print(f"  ✨ Best model saved!")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    # =============================================================================
    # 7. 测试阶段 (新增)
    # =============================================================================
    print("\n" + "="*30 + " Final Testing " + "="*30)
    print(f"Loading best student model from: {ckpt_path}")
    
    # 加载最佳权重
    checkpoint = torch.load(ckpt_path, map_location=cfg.device, weights_only=True)
    student.load_state_dict(checkpoint['model_state'])
    student.eval()

    # 在测试集上评估
    test_metrics = evaluate_with_teacher(cfg, teacher, student, test_loader, criterion)

    print(f"\n✅ Test Result (Student):")
    print(f"   Accuracy : {test_metrics['s_acc']:.4f}")
    print(f"   F1 Score : {test_metrics['s_f1']:.4f}")
    print(f"   Loss     : {test_metrics['loss']:.4f}")
    print(f"   Loss Details: CE={test_metrics['l_ce']:.4f}, Graph={test_metrics['l_graph']:.4f}, Temporal={test_metrics['l_temporal']:.4f}, Logits={test_metrics['l_logits']:.4f}")
    
    print(f"\n📊 Teacher EEG Branch Reference (On Test Set):")
    print(f"   Accuracy : {test_metrics['t_acc']:.4f}")

    print(f"\n📝 Confusion Matrix (Student):")
    print(test_metrics['cm'])
    print("="*75 + "\n")

# =============================================================================
# 辅助评估函数
# =============================================================================

@torch.no_grad()
def evaluate_with_teacher(cfg, teacher, student, loader, criterion, epoch=0):
    student.eval()
    teacher.eval()
    
    loss_meter = AverageMeter()
    ce_meter = AverageMeter()
    graph_meter = AverageMeter()
    temporal_meter = AverageMeter()
    logits_meter = AverageMeter()
    s_preds, t_preds, labels = [], [], []

    for eeg, pcc, audio, vision, y in loader:
        eeg, pcc, audio, vision, y = [d.to(cfg.device) for d in [eeg, pcc, audio, vision, y]]
        
        t_out = teacher(eeg, pcc, audio, vision)
        s_out = student(eeg, pcc)
        
        losses = criterion(s_out, t_out, y, epoch)
        loss_meter.update(losses['loss'].item(), eeg.size(0))
        ce_meter.update(losses['l_ce'].item(), eeg.size(0))
        graph_meter.update(losses['l_graph'].item(), eeg.size(0))
        temporal_meter.update(losses['l_temporal'].item(), eeg.size(0))
        logits_meter.update(losses['l_logits'].item(), eeg.size(0))

        s_preds.extend(s_out['logits'].argmax(dim=-1).cpu().numpy())
        t_preds.extend(t_out['eeg_logits'].argmax(dim=-1).cpu().numpy())
        labels.extend(y.cpu().numpy())

    return {
        'loss':  loss_meter.avg,
        'l_ce': ce_meter.avg,
        'l_graph': graph_meter.avg,
        'l_temporal': temporal_meter.avg,
        'l_logits': logits_meter.avg,
        's_acc': accuracy_score(labels, s_preds),
        's_f1':  f1_score(labels, s_preds, average='weighted'),
        't_acc': accuracy_score(labels, t_preds),
        'cm':    confusion_matrix(labels, s_preds)
    }

if __name__ == '__main__':
    main()