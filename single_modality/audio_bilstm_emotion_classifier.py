import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import random

# ==========================================
# 0. 设置随机种子以确保实验可复现
# ==========================================
def set_seed(seed=2024):
    """设置所有随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时
    torch.backends.cudnn.deterministic = True  # 确保卷积操作确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动优化以确保可复现
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🎲 随机种子已设置为: {seed}")

SEED = 2024

# ==========================================
# 1. z-score归一化
# ==========================================
def z_score_normalize(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray):
    """
    对特征进行z-score归一化
    使用训练数据的均值和标准差来归一化所有数据
    """
    # 计算训练数据的均值和标准差 (按特征维度)
    # train_x shape: [samples, frames, features]
    mean = np.mean(train_x, axis=(0, 1), keepdims=True)  # [1, 1, features]
    std = np.std(train_x, axis=(0, 1), keepdims=True)    # [1, 1, features]
    
    # 避免除零错误
    std[std == 0] = 1.0
    
    print(f"📊 归一化统计: 均值范围 [{mean.min():.4f}, {mean.max():.4f}], 标准差范围 [{std.min():.4f}, {std.max():.4f}]")
    
    # 对所有数据进行归一化
    norm_train = (train_x - mean) / std
    norm_val = (val_x - mean) / std
    norm_test = (test_x - mean) / std
    
    print("🧹 已完成z-score归一化")
    return norm_train, norm_val, norm_test

# ==========================================
# 2. 数据加载器 (适配OpenSmile特征)
# ==========================================
class CrossSubjectAudioDataset:
    def __init__(self, data_root: str, num_subjects: int = 42, random_seed: int = SEED):
        self.data_root = data_root
        self.num_subjects = num_subjects
        self._np_rng = np.random.RandomState(random_seed)
        
        self.subjects = list(range(1, num_subjects + 1))
        
        # 使用固定的subject划分 (与OpenFace相同)
        self.train_subjects = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 18, 19, 24, 25, 28, 29, 33, 34, 35, 36, 37, 39, 40, 42]
        self.val_subjects = [10, 14, 15, 16, 20, 22, 31, 32]
        self.test_subjects = [1, 17, 21, 23, 26, 27, 30, 38, 41]
        
        # 打印划分的subject
        print("=" * 60)
        print("📊 被试划分情况:")
        print(f"训练集被试 (Train): {self.train_subjects}")
        print(f"验证集被试 (Val):   {self.val_subjects}")
        print(f"测试集被试 (Test):  {self.test_subjects}")
        print(f"总计: {len(self.train_subjects)} 训练, {len(self.val_subjects)} 验证, {len(self.test_subjects)} 测试")
        print("=" * 60)

    def _load_subject_data(self, subject_id):
        """加载单个subject的OpenSmile特征数据"""
        file_path = os.path.join(self.data_root, f"subject_{subject_id:02d}_aud_egemaps_32f_raw.pkl")
        
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            return None, None, None, None
        
        with open(file_path, 'rb') as f:
            # 格式: [train_x, train_y, test_x, test_y]
            train_x, train_y, test_x, test_y = pickle.load(f)
        
        return train_x, train_y, test_x, test_y

    def load_data_by_split(self, split='train'):
        """按划分加载数据"""
        mapping = {'train': self.train_subjects, 'val': self.val_subjects, 'test': self.test_subjects}
        
        all_x = []
        all_y = []
        
        for sid in mapping[split]:
            train_x, train_y, test_x, test_y = self._load_subject_data(sid)
            
            if train_x is None:
                continue
            
            # 合并train和test数据
            combined_x = np.concatenate([train_x, test_x], axis=0)
            combined_y = np.concatenate([train_y, test_y], axis=0)
            
            all_x.append(combined_x)
            all_y.append(combined_y)
        
        # 合并所有subject的数据
        if len(all_x) > 0:
            final_x = np.concatenate(all_x, axis=0)
            final_y = np.concatenate(all_y, axis=0)
        else:
            final_x = np.array([])
            final_y = np.array([])
        
        return final_x, final_y

# ==========================================
# 3. 模型定义 (Bi-LSTM)
# ==========================================
class AudioBiLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        out, (h_n, _) = self.lstm(x)
        # 取双向LSTM最后一层隐藏状态
        # h_n: [num_layers * 2, batch, hidden_dim]
        feat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.fc(feat)

# ==========================================
# 4. 数据集类
# ==========================================
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 5. 训练函数
# ==========================================
def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        if is_train:
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            outputs = model(x)
            loss = criterion(outputs, y)
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

# ==========================================
# 6. 主程序
# ==========================================
def main():
    # 设置随机种子
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # OpenSmile特征路径
    data_root = '/data2/mingzhi/BCI/dataset/EAV_old/Audio_OpenSmile_32frame_raw'
    
    # 创建数据管理器
    ds_manager = CrossSubjectAudioDataset(data_root)
    
    print("\n🚀 正在从磁盘加载 OpenSmile 特征...")
    tr_x_raw, tr_y = ds_manager.load_data_by_split('train')
    val_x_raw, val_y = ds_manager.load_data_by_split('val')
    test_x_raw, test_y = ds_manager.load_data_by_split('test')
    
    print(f"📊 数据加载完成:")
    print(f"   训练集: {tr_x_raw.shape}, 标签: {tr_y.shape}")
    print(f"   验证集: {val_x_raw.shape}, 标签: {val_y.shape}")
    print(f"   测试集: {test_x_raw.shape}, 标签: {test_y.shape}")
    
    # 应用z-score归一化
    tr_x, val_x, test_x = z_score_normalize(tr_x_raw, val_x_raw, test_x_raw)
    
    # 获取输入维度
    # tr_x shape: [samples, frames, features]
    input_dim = tr_x.shape[2]  # 25 (eGeMAPS LLD特征数)
    num_classes = len(np.unique(tr_y))
    
    print(f"\n📊 模型配置:")
    print(f"   输入维度: {input_dim}")
    print(f"   序列长度: {tr_x.shape[1]} (帧)")
    print(f"   类别数: {num_classes}")
    
    # 创建DataLoader
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(AudioDataset(tr_x, tr_y), batch_size=64, shuffle=True, generator=g)
    val_loader = DataLoader(AudioDataset(val_x, val_y), batch_size=64, shuffle=False)
    test_loader = DataLoader(AudioDataset(test_x, test_y), batch_size=64, shuffle=False)
    
    # 创建模型
    model = AudioBiLSTMEmotionClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)
    
    print(f"\n🧠 模型结构:")
    print(model)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练配置
    best_acc = 0
    best_epoch = 0
    save_name = 'best_opensmile_bilstm.pth'
    patience = 10
    epochs_without_improvement = 0
    
    print(f"\n🚀 开始训练...")
    print("=" * 60)
    
    for epoch in range(100):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, is_train=False)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_name)
        else:
            epochs_without_improvement += 1
        
        print(f"Epoch {epoch+1:02d} | Loss: {tr_loss:.4f} | Tr: {tr_acc:.2f}% | Val: {val_acc:.2f}%"
              + (" ✅" if val_acc == best_acc else f" ({epochs_without_improvement}/{patience})"))
        
        # 早停检查
        if epochs_without_improvement >= patience:
            print(f"\n🛑 早停触发！验证集准确率 {patience} 个epoch无改善")
            break
    
    print(f"\n🏆 最佳 Val Acc: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"💾 模型已保存至: {save_name}")
    
    # 加载最佳模型并在测试集上评估
    print("\n🔍 在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(save_name))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_train=False)
    print(f"📊 测试集结果: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
