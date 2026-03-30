import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from multimodal.model.Teacher import TeacherModel
from dataset.dataset import CrossSubjectMultiModalDataset, DataLoader


def visualize_S_attn():
    """
    可视化Teacher模型的S_attn空间注意力矩阵
    """
    print("🔍 开始可视化S_attn...")
    
    # 设置随机种子
    torch.manual_seed(2024)
    np.random.seed(2024)
    
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 数据路径
    data_root_dict = {
        'eeg'   : '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'audio' : '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace',
    }
    
    # 1. 加载数据
    print("\n📥 加载数据...")
    manager = CrossSubjectMultiModalDataset(
        data_root_dict,
        audio_feature_type='opensmile',
        vision_feature_type='openface'
    )
    
    (tr_eeg, tr_y, tr_pcc), (va_eeg, va_y, va_pcc), (te_eeg, te_y, te_pcc) = \
        manager.get_all_splits('eeg', extract_de=True, normalize=True, compute_pcc=True)
    
    (tr_aud, _), (va_aud, _), (te_aud, _) = manager.get_all_splits('audio', normalize=True)
    (tr_vis, _), (va_vis, _), (te_vis, _) = manager.get_all_splits('vision', normalize=True)
    
    # 创建对齐数据集
    class AlignedDataset(torch.utils.data.Dataset):
        def __init__(self, eeg, pcc, labels, audio, vision):
            self.eeg = eeg
            self.pcc = pcc
            self.labels = labels
            self.audio = audio
            self.vision = vision
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return (self.eeg[idx], self.pcc[idx], self.audio[idx], 
                    self.vision[idx], self.labels[idx])
    
    # 只取前几个样本进行可视化
    num_samples = 4
    test_dataset = AlignedDataset(
        te_eeg[:num_samples], te_pcc[:num_samples], te_y[:num_samples],
        te_aud[:num_samples], te_vis[:num_samples]
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 初始化Teacher模型
    print("\n🤖 初始化Teacher模型...")
    teacher = TeacherModel(
        audio_input_dim=25,
        vision_input_dim=161,
        av_hidden_dim=64,
        av_num_layers=1,
        av_dropout=0.5,
        num_nodes=30,
        eeg_in_features=5,
        gcn_hidden=64,
        gcn_out=64,
        lstm_hidden=64,
        lstm_layers=1,
        eeg_dropout=0.5,
        dk=32,
        fc_hidden=64,
        num_classes=5,
    ).to(device)
    
    # 尝试加载预训练权重
    ckpt_path = './checkpoints/best_teacher.pth'
    if os.path.exists(ckpt_path):
        print(f"📦 加载预训练权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt['model_state'])
    else:
        print("⚠️  未找到预训练权重，使用随机初始化")
    
    teacher.eval()
    
    # 3. 前向传播并获取S_attn
    print("\n🔄 前向传播...")
    S_attn_list = []
    labels_list = []
    
    with torch.no_grad():
        for eeg, pcc, audio, vision, y in test_loader:
            eeg, pcc, audio, vision, y = [d.to(device) for d in [eeg, pcc, audio, vision, y]]
            
            outputs = teacher(eeg, pcc, audio, vision)
            S_attn = outputs['S_attn']
            
            S_attn_list.append(S_attn.cpu().numpy())
            labels_list.append(y.cpu().numpy())
    
    # 4. 可视化S_attn
    print("\n📊 生成可视化...")
    os.makedirs('./visualizations', exist_ok=True)
    
    for sample_idx in range(len(S_attn_list)):
        S_attn = S_attn_list[sample_idx]  # [1, T, N, N]
        label = labels_list[sample_idx][0]
        
        B, T, N, _ = S_attn.shape
        
        # 创建热力图
        fig, axes = plt.subplots(2, min(T, 5), figsize=(15, 8))
        if T == 1:
            axes = axes.reshape(2, 1)
        
        # 标题
        fig.suptitle(f'Sample {sample_idx + 1}, Label: {label}', fontsize=16)
        
        # 可视化前min(T, 5)个时间步
        for t in range(min(T, 5)):
            # 原始注意力矩阵
            ax1 = axes[0, t]
            im1 = ax1.imshow(S_attn[0, t], cmap='viridis', aspect='equal')
            ax1.set_title(f'Time {t + 1}')
            ax1.set_xlabel('Node')
            ax1.set_ylabel('Node')
            plt.colorbar(im1, ax=ax1)
            
            # 行求和（每行是一个节点对其他节点的注意力）
            row_sum = S_attn[0, t].sum(axis=1)
            ax2 = axes[1, t]
            ax2.bar(range(N), row_sum)
            ax2.set_title(f'Row Sum (Time {t + 1})')
            ax2.set_xlabel('Node')
            ax2.set_ylabel('Attention Sum')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f'./visualizations/S_attn_sample_{sample_idx + 1}_label_{label}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 已保存: {save_path}")
        plt.close()
        
        # 另外，可视化同一时间步的多个样本的平均
        if sample_idx == 0:
            # 只对第一个样本进行时间步平均
            avg_S_attn = S_attn[0].mean(axis=0)  # [N, N]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 平均注意力矩阵
            im1 = ax1.imshow(avg_S_attn, cmap='viridis', aspect='equal')
            ax1.set_title(f'Average S_attn (Sample {sample_idx + 1})')
            ax1.set_xlabel('Node')
            ax1.set_ylabel('Node')
            plt.colorbar(im1, ax=ax1)
            
            # 节点重要性（行平均）
            node_importance = avg_S_attn.mean(axis=1)
            ax2.bar(range(N), node_importance)
            ax2.set_title('Node Importance (Avg Attention)')
            ax2.set_xlabel('Node')
            ax2.set_ylabel('Average Attention')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = f'./visualizations/S_attn_avg_sample_{sample_idx + 1}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 已保存: {save_path}")
            plt.close()
    
    print("\n🎉 可视化完成! 结果保存在 ./visualizations/ 目录")


if __name__ == '__main__':
    visualize_S_attn()
