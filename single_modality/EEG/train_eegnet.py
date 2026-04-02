import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import random

# 添加项目根目录到Python路径
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

# 导入EEGNet模型
class EEGNet(nn.Module):
    def __init__(self, num_classes=3, num_channels=30, num_samples=1000, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        # 1. 卷积层保持不变...
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # 2. 自动计算全连接层输入维度
        # 创建一个 dummy input 来推断输出尺寸
        dummy_input = torch.zeros(1, 1, num_channels, num_samples)
        with torch.no_grad():
            x = self.first_conv(dummy_input)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            flatten_dim = x.numel() # 自动计算展平后的总元素数

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, num_classes)
        )

# 导入数据集加载器
from dataset.dataset import create_single_modality_dataloaders

# ==========================================
# 配置
# ==========================================
SEED = 2024
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 10  # 早停耐心值

# 数据路径
DATA_ROOT_DICT = {
    'eeg': '/data2/mingzhi/BCI/dataset/EAV_old/EEG'
}

# ==========================================
# 设置随机种子
# ==========================================
def set_seed(seed):
    """设置所有随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🎲 随机种子已设置为: {seed}")

# ==========================================
# 训练函数
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==========================================
# 验证函数
# ==========================================
def validate_epoch(model, loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ==========================================
# 主函数
# ==========================================
def main():
    # 设置随机种子
    set_seed(SEED)
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 使用设备: {device}")
    
    # 加载数据（使用原始EEG特征和归一化）
    print("\n🚀 加载原始EEG特征数据...")
    train_loader, val_loader, test_loader = create_single_modality_dataloaders(
        segment_1s=False,
        data_root_dict=DATA_ROOT_DICT,
        modality='eeg',
        batch_size=BATCH_SIZE,
        extract_de=False,  # 使用原始EEG特征
        normalize=True    # 进行z-score归一化
    )
    
    # 检查数据形状
    for x, y in train_loader:
        print(f"📊 输入数据形状: {x.shape}")
        print(f"📊 输入数据类型: {x.dtype}")
        print(f"📊 标签形状: {y.shape}")
        print(f"📊 类别数: {len(torch.unique(y))}")
        break
    
    # 获取输入维度
    input_channels = x.shape[1]
    input_features = x.shape[2]
    num_classes = len(torch.unique(y))
    
    print(f"\n🎯 模型配置:")
    print(f"   输入通道数: {input_channels}")
    print(f"   输入特征数: {input_features}")
    print(f"   类别数: {num_classes}")
    
    # 创建EEGNet模型
    model = EEGNet(
        num_classes=num_classes,
        num_channels=input_channels,
        num_samples=input_features,
        dropout_rate=0.5
    ).to(device)
    
    print(f"\n🧠 模型结构:")
    print(model)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # 训练配置
    best_val_acc = 0
    best_epoch = 0
    epochs_without_improvement = 0
    save_path = f'eegnet_de_best.pth'
    
    print(f"\n🚀 开始训练...")
    print("=" * 80)
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"📈 Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ✅")
        else:
            epochs_without_improvement += 1
            print(f"📉 Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% ({epochs_without_improvement}/{PATIENCE})")
        
        # 早停检查
        if epochs_without_improvement >= PATIENCE:
            print(f"\n🛑 早停触发！验证集准确率 {PATIENCE} 个epoch无改善")
            break
    
    print(f"\n🏆 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"💾 最佳模型已保存至: {save_path}")
    
    # 测试最佳模型
    print("\n🔍 在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"📊 测试集结果: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
