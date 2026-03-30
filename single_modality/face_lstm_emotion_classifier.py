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
# 1. 特征配置 (适配新的 OpenFace 特征)
# 根据extract_openface_features.py的特征结构：
# 8维: 眼睛注视方向 Gaze (gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y)
# 136维: 68个面部关键点 (x0...x67, y0...y67)
# 17维: 17个面部动作单元 AU (AU01_r...AU45_r) 
# 总计: 161维
# ==========================================
FEATURE_CONFIG = {
    'all':           None,        # 使用全部特征
    'au':            'AU',        # AU特征 (17维)
    'gaze':          'gaze_',     # 视线特征 (8维)
    'landmark':      ['x_', 'y_'], # 2D关键点 (136维)
    'au_gaze':       ['AU', 'gaze_'],      # AU + gaze (25维)
    'au_landmark':   ['AU', 'x_', 'y_'],   # AU + landmark (153维)
    'gaze_landmark': ['gaze_', 'x_', 'y_'], # gaze + landmark (144维)
    'au_gaze_landmark': ['AU', 'gaze_', 'x_', 'y_'], # 全部特征 (161维)
}

USE_FEATURE = 'au_gaze_landmark'  # 默认使用全部特征

def get_feature_indices(feature_dim, target='all'):
    """
    根据特征维度和目标类型获取特征索引
    基于extract_openface_features.py的特征结构
    实际特征顺序：gaze_ (8维) + x_ (68维) + y_ (68维) + AU_ (17维) = 161维
    """
    # 特征分布
    gaze_end = 8       # 8个gaze特征
    x_end = gaze_end + 68  # 68个x坐标
    y_end = x_end + 68  # 68个y坐标
    au_end = y_end + 17  # 17个AU特征
    
    indices = []
    
    if target == 'all':
        indices = list(range(feature_dim))
    elif target == 'au':
        indices = list(range(y_end, au_end))
    elif target == 'gaze':
        indices = list(range(0, gaze_end))
    elif target == 'landmark':
        indices = list(range(gaze_end, y_end))  # x_ + y_
    elif target == 'au_gaze':
        indices.extend(list(range(0, gaze_end)))  # gaze
        indices.extend(list(range(y_end, au_end)))  # AU
    elif target == 'au_landmark':
        indices.extend(list(range(gaze_end, y_end)))  # x_ + y_
        indices.extend(list(range(y_end, au_end)))  # AU
    elif target == 'gaze_landmark':
        indices.extend(list(range(0, gaze_end)))  # gaze
        indices.extend(list(range(gaze_end, y_end)))  # x_ + y_
    elif target == 'au_gaze_landmark':
        indices.extend(list(range(0, gaze_end)))  # gaze
        indices.extend(list(range(gaze_end, y_end)))  # x_ + y_
        indices.extend(list(range(y_end, au_end)))  # AU
    
    print(f"🔍 提取子集 [{target}]: 选中 {len(indices)} 列")
    return indices

def remove_outliers(x_list: List[np.ndarray], threshold=1000):
    """
    移除特征中的异常值
    """
    cleaned_x = []
    for x in x_list:
        x_clean = np.where(np.abs(x) > threshold, 0, x)
        cleaned_x.append(x_clean)
    print(f"🧹 已清理异常值，阈值: {threshold}")
    return cleaned_x

def slice_openface_features(x_list: List[np.ndarray], target='all'):
    """
    根据目标类型切分特征
    """
    if target == 'all':
        return x_list
    if x_list:
        feature_dim = x_list[0].shape[1]
        indices = get_feature_indices(feature_dim, target)
        sliced_x = [x[:, indices] for x in x_list]
        cleaned_x = remove_outliers(sliced_x)
        return cleaned_x
    else:
        return x_list

def z_score_normalize(train_x: List[np.ndarray], val_x: List[np.ndarray], test_x: List[np.ndarray]):
    """
    对特征进行z-score归一化
    使用训练数据的均值和标准差来归一化所有数据
    """
    # 计算训练数据的均值和标准差
    # 将所有训练数据连接成一个大数组
    train_concat = np.concatenate(train_x, axis=0)
    mean = np.mean(train_concat, axis=0)
    std = np.std(train_concat, axis=0)
    
    # 避免除零错误
    std[std == 0] = 1.0
    
    print(f"📊 归一化统计: 均值范围 [{mean.min():.4f}, {mean.max():.4f}], 标准差范围 [{std.min():.4f}, {std.max():.4f}]")
    
    # 对训练、验证和测试数据进行归一化
    def normalize(x_list):
        return [(x - mean) / std for x in x_list]
    
    norm_train = normalize(train_x)
    norm_val = normalize(val_x)
    norm_test = normalize(test_x)
    
    print("🧹 已完成z-score归一化")
    return norm_train, norm_val, norm_test

# ==========================================
# 2. 数据加载器 (适配新的 OpenFace 路径)
# ==========================================
class CrossSubjectMultiModalDataset:
    def __init__(self, data_root_dict: Dict[str, str], num_subjects: int = 42, random_seed: int = SEED):
        self.data_root_dict = data_root_dict
        self.num_subjects = num_subjects
        self._np_rng = np.random.RandomState(random_seed)

        # 文件名后缀映射
        self.file_suffix_map = {'openface': '_vis_openface.pkl'}
        
        self.subjects = list(range(1, num_subjects + 1))
        
        # 使用固定的subject划分
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

    def _split_subjects_default(self):
        all_subs = list(self.subjects)
        self._np_rng.shuffle(all_subs)
        n_tr, n_val = int(len(all_subs) * 0.6), int(len(all_subs) * 0.2)
        return sorted(all_subs[:n_tr]), sorted(all_subs[n_tr:n_tr+n_val]), sorted(all_subs[n_tr+n_val:])

    def _load_subject_data(self, subject_id, modality):
        suffix = self.file_suffix_map.get(modality, f'_{modality}.pkl')
        file_path = os.path.join(self.data_root_dict[modality], f"subject_{subject_id:02d}{suffix}")
        
        if not os.path.exists(file_path):
            return [], []
        
        with open(file_path, 'rb') as f:
            # 新的提取脚本保存格式是: [train_feats, train_y, test_feats, test_y]
            data = pickle.load(f)
        
        tr_x, tr_y, te_x, te_y = data
        return tr_x + te_x, list(tr_y) + list(te_y)

    def load_data_by_split(self, split='train', modality='openface'):
        mapping = {'train': self.train_subjects, 'val': self.val_subjects, 'test': self.test_subjects}
        x_all, y_all = [], []
        for sid in mapping[split]:
            x, y = self._load_subject_data(sid, modality)
            x_all.extend(x)
            y_all.extend(y)
        return x_all, y_all

# ==========================================
# 3. 模型与训练 (保持不变，仅修改 input_dim)
# ==========================================
class FaceLSTMEmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        # 取双向 LSTM 最后一层隐藏状态
        feat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.fc(feat)

class OpenFaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if is_train: optimizer.zero_grad()
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
# 5. 主程序
# ==========================================
def main():
    # 设置随机种子
    set_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 更新路径指向新的OpenFace结果目录
    data_root = {'openface': '/data2/mingzhi/BCI/dataset/EAV_old/Vision_OpenFace'}

    ds_manager = CrossSubjectMultiModalDataset(data_root)

    print("🚀 正在从磁盘加载 OpenFace 特征...")
    tr_x_raw,  tr_y  = ds_manager.load_data_by_split('train', 'openface')
    val_x_raw, val_y = ds_manager.load_data_by_split('val',   'openface')
    test_x_raw, test_y = ds_manager.load_data_by_split('test',  'openface')

    # 应用特征选择
    tr_x = slice_openface_features(tr_x_raw, USE_FEATURE)
    val_x = slice_openface_features(val_x_raw, USE_FEATURE)
    test_x = slice_openface_features(test_x_raw, USE_FEATURE)

    # 应用z-score归一化
    tr_x, val_x, test_x = z_score_normalize(tr_x, val_x, test_x)

    input_dim = tr_x[0].shape[1]
    num_classes = len(np.unique(tr_y))
    print(f"📊 特征模式: {USE_FEATURE} | 最终输入维度: {input_dim} | 类别数: {num_classes}")

    # 使用Generator确保DataLoader的可复现性
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(OpenFaceDataset(tr_x,  tr_y),  batch_size=64, shuffle=True, generator=g)
    val_loader   = DataLoader(OpenFaceDataset(val_x, val_y), batch_size=64, shuffle=False)
    test_loader  = DataLoader(OpenFaceDataset(test_x, test_y), batch_size=64, shuffle=False)

    model     = FaceLSTMEmotionClassifier(input_dim, 128, 2, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    save_name = f'best_openface_lstm_{USE_FEATURE}.pth'
    
    # 早停设置
    patience = 10  # 容忍的epoch数
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(60):
        tr_loss, tr_acc   = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

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
            print(f"   最佳epoch: {best_epoch}, 最佳验证准确率: {best_acc:.2f}%")
            break

    print(f"\n🏆 最佳 Val Acc: {best_acc:.2f}% (Epoch {best_epoch})  已保存至 {save_name}")
    
    # 加载最佳模型并在测试集上评估
    print("\n🔍 在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(save_name))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_train=False)
    print(f"📊 测试集结果: Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()