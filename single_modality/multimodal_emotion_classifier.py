import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class OpenFaceOpenSmileDataset(Dataset):
    """
    整合 OpenFace 和 OpenSmile 特征的数据集
    """
    def __init__(self, data_dir, subject_ids, split='train'):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            subject_ids: 被试ID列表
            split: 数据集分割类型 ('train' 或 'test')
        """
        self.data_dir = data_dir
        self.subject_ids = subject_ids
        self.split = split
        self.openface_features = []
        self.opensmile_features = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """
        加载数据
        """
        for subject_id in self.subject_ids:
            # 加载 OpenFace 特征
            openface_file = os.path.join(self.data_dir, 'Vision_OpenFace', f"subject_{subject_id:02d}_vis_openface.pkl")
            # 加载 OpenSmile 特征
            opensmile_file = os.path.join(self.data_dir, 'Audio_OpenSmile', f"subject_{subject_id:02d}_aud_opensmile.pkl")
            
            if not os.path.exists(openface_file):
                print(f"OpenFace 文件不存在: {openface_file}")
                continue
            
            if not os.path.exists(opensmile_file):
                print(f"OpenSmile 文件不存在: {opensmile_file}")
                continue
            
            # 加载 OpenFace 数据
            with open(openface_file, 'rb') as f:
                openface_data = pickle.load(f)
            
            # 加载 OpenSmile 数据
            with open(opensmile_file, 'rb') as f:
                opensmile_data = pickle.load(f)
            
            # 提取对应分割的数据
            if self.split == 'train':
                openface_subject_features = openface_data[0]  # train_x
                opensmile_subject_features = opensmile_data[0]  # train_x
                subject_labels = openface_data[1]  # train_y
            else:
                openface_subject_features = openface_data[2]  # test_x
                opensmile_subject_features = opensmile_data[2]  # test_x
                subject_labels = openface_data[3]  # test_y
            
            # 确保特征数量匹配
            min_len = min(len(openface_subject_features), len(opensmile_subject_features), len(subject_labels))
            
            # 添加到数据集
            for i in range(min_len):
                # 对 OpenFace 特征取时间维度的平均值
                openface_feature = np.mean(openface_subject_features[i], axis=0)
                # OpenSmile 特征已经是时间维度的平均值
                opensmile_feature = opensmile_subject_features[i]
                
                self.openface_features.append(openface_feature)
                self.opensmile_features.append(opensmile_feature)
                self.labels.append(subject_labels[i])
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            openface_feature: OpenFace 特征张量
            opensmile_feature: OpenSmile 特征张量
            label: 标签张量
        """
        openface_feature = torch.tensor(self.openface_features[idx], dtype=torch.float32)
        opensmile_feature = torch.tensor(self.opensmile_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return openface_feature, opensmile_feature, label

class MultiModalEmotionClassifier(nn.Module):
    """
    多模态情绪分类器
    """
    def __init__(self, openface_dim, opensmile_dim, hidden_dim, num_classes):
        """
        初始化模型
        
        Args:
            openface_dim: OpenFace 特征维度
            opensmile_dim: OpenSmile 特征维度
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
        """
        super(MultiModalEmotionClassifier, self).__init__()
        
        # OpenFace 特征处理
        self.openface_fc = nn.Sequential(
            nn.Linear(openface_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # OpenSmile 特征处理
        self.opensmile_fc = nn.Sequential(
            nn.Linear(opensmile_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 多模态融合
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, openface_feature, opensmile_feature):
        """
        前向传播
        
        Args:
            openface_feature: OpenFace 特征张量
            opensmile_feature: OpenSmile 特征张量
        
        Returns:
            out: 输出张量
        """
        # 处理各模态特征
        openface_out = self.openface_fc(openface_feature)
        opensmile_out = self.opensmile_fc(opensmile_feature)
        
        # 融合特征
        fused = torch.cat((openface_out, opensmile_out), dim=1)
        out = self.fusion_fc(fused)
        
        return out

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
    
    Returns:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
    """
    model.train()
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for openface_features, opensmile_features, labels in progress_bar:
            openface_features = openface_features.to(device)
            opensmile_features = opensmile_features.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(openface_features, opensmile_features)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 计算损失和准确率
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix(loss=running_loss/(len(progress_bar)), acc=100.*correct/total)
        
        # 计算 epoch 损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    return train_losses, train_accs

def test_model(model, test_loader, criterion, device):
    """
    测试模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        test_loss: 测试损失
        test_acc: 测试准确率
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for openface_features, opensmile_features, labels in tqdm(test_loader, desc='Testing'):
            openface_features = openface_features.to(device)
            opensmile_features = opensmile_features.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(openface_features, opensmile_features)
            loss = criterion(outputs, labels)
            
            # 计算损失和准确率
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # 计算平均损失和准确率
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    return test_loss, test_acc

def plot_results(train_losses, train_accs, test_loss=None, test_acc=None):
    """
    绘制训练结果
    
    Args:
        train_losses: 训练损失列表
        train_accs: 训练准确率列表
        test_loss: 测试损失
        test_acc: 测试准确率
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Train Loss')
    if test_loss is not None:
        ax1.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    
    # 绘制准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    if test_acc is not None:
        ax2.axhline(y=test_acc, color='r', linestyle='--', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('multimodal_training_results.png')
    print('训练结果已保存到 multimodal_training_results.png')

class CrossSubjectMultiModalDataset:
    """
    Cross-Subject Multi-Modal Dataset Loader for BCI Data
    """
    def __init__(self, num_subjects=42, random_seed=2024):
        self.num_subjects = num_subjects
        self.random_seed = random_seed
        
        # RNGs
        self._np_rng = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
        
        # 默认被试列表
        self.subjects = list(range(1, self.num_subjects + 1))
        
        # 默认分割
        self.train_subjects, self.val_subjects, self.test_subjects = self._split_subjects_default()
        
        print(f"Train subjects ({len(self.train_subjects)}): {self.train_subjects}")
        print(f"Val subjects   ({len(self.val_subjects)}): {self.val_subjects}")
        print(f"Test subjects  ({len(self.test_subjects)}): {self.test_subjects}")
    
    def _split_subjects_default(self):
        """Split subjects into train/val/test sets with 6:2:2 ratio (random shuffle)."""
        all_subjects = list(self.subjects)
        self._np_rng.shuffle(all_subjects)

        n_train = int(self.num_subjects * 0.6)
        n_val = int(self.num_subjects * 0.2)

        train_subjects = all_subjects[:n_train]
        val_subjects = all_subjects[n_train:n_train + n_val]
        test_subjects = all_subjects[n_train + n_val:]

        return sorted(train_subjects), sorted(val_subjects), sorted(test_subjects)

def main():
    """
    主函数
    """
    # 配置参数
    data_dir = "/data2/mingzhi/BCI/dataset/EAV_old"
    
    # 超参数
    openface_dim = 0  # 将在加载数据后确定
    opensmile_dim = 0  # 将在加载数据后确定
    hidden_dim = 256
    num_classes = 5
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 使用 CrossSubjectMultiModalDataset 来获取被试分割
    dataset = CrossSubjectMultiModalDataset(num_subjects=42)
    
    # 加载训练数据
    print("加载训练数据...")
    train_dataset = OpenFaceOpenSmileDataset(data_dir, dataset.train_subjects, split='train')
    print(f"训练数据大小: {len(train_dataset)}")
    
    # 确定特征维度
    if len(train_dataset) > 0:
        openface_sample, opensmile_sample, _ = train_dataset[0]
        openface_dim = openface_sample.shape[0]
        opensmile_dim = opensmile_sample.shape[0]
        print(f"OpenFace 特征维度: {openface_dim}")
        print(f"OpenSmile 特征维度: {opensmile_dim}")
    else:
        print("训练数据为空，请检查数据目录")
        return
    
    # 加载测试数据
    print("加载测试数据...")
    test_dataset = OpenFaceOpenSmileDataset(data_dir, dataset.test_subjects, split='test')
    print(f"测试数据大小: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = MultiModalEmotionClassifier(openface_dim, opensmile_dim, hidden_dim, num_classes).to(device)
    print(f"模型结构: {model}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练...")
    train_losses, train_accs = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    
    # 测试模型
    print("开始测试...")
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    
    # 绘制结果
    plot_results(train_losses, train_accs, test_loss, test_acc)
    
    # 保存模型
    torch.save(model.state_dict(), 'multimodal_emotion_classifier.pth')
    print('模型已保存到 multimodal_emotion_classifier.pth')

if __name__ == "__main__":
    main()
