import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI')
import os
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x): # note we carefully add the positional encoding, omitted
        x = x #+ self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x

class EEGFormer(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 5),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        # x = x.permute(0, 2, 1)
        # x = x.mean(dim=-1)
        # x = self.mlp(x)
        return x


class CrossSubjectEEGNetTrainer:
    """跨被试EEGNet训练器"""
    
    def __init__(self, 
                 data_root_dict,
                 num_classes=5,
                 Chans=64,
                 Samples=128,
                 dropoutRate=0.5,
                 kernLength=64,
                 F1=8,
                 D=2,
                 F2=16,
                 lr=0.001,
                 batch_size=64,
                 num_workers=4,
                 random_seed=2024,
                 subject_csv_path: Optional[str] = None,
                 checkpoint_dir='eeg_checkpoints'):
        """
        Args:
            data_root_dict: 数据根目录字典
            num_classes: 分类类别数
            Chans: EEG通道数
            Samples: 每个样本的时间点数
            dropoutRate: Dropout率
            kernLength: 第一层卷积核长度
            F1: 第一层卷积滤波器数
            D: 深度乘数
            F2: 第二层卷积滤波器数
            lr: 学习率
            batch_size: 批大小
            num_workers: DataLoader工作进程数
            random_seed: 随机种子
            subject_csv_path: 被试CSV文件路径
            checkpoint_dir: checkpoint保存目录
        """
        print("="*80)
        print("Initializing Cross-Subject EEGNet Classifier")
        print("="*80)
        
        self.num_classes = num_classes
        self.Chans = Chans
        self.Samples = Samples
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建checkpoint目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        # 导入数据集
        from dataset.dataset import CrossSubjectMultiModalDataset, MultiModalTensorDataset
        
        
        # 初始化跨被试数据集
        print("\nLoading cross-subject dataset...")
        self.dataset_loader = CrossSubjectMultiModalDataset(
            data_root_dict=data_root_dict,
            random_seed=random_seed,
            subject_csv_path=subject_csv_path
        )
        
        # 加载EEG模态数据
        print("\nLoading EEG modality data for all splits...")
        train_data_dict = self.dataset_loader.load_all_modalities_by_split('train')
        val_data_dict = self.dataset_loader.load_all_modalities_by_split('val')
        test_data_dict = self.dataset_loader.load_all_modalities_by_split('test')
        
        # 创建PyTorch Datasets
        self.train_dataset = MultiModalTensorDataset(train_data_dict)
        self.val_dataset = MultiModalTensorDataset(val_data_dict)
        self.test_dataset = MultiModalTensorDataset(test_data_dict)
        
        # 从第一个样本获取实际的数据维度
        sample = self.train_dataset[0]
        eeg_shape = sample['eeg'].shape  # (Chans, Samples)
        self.Chans = eeg_shape[0]
        self.Samples = eeg_shape[1]
        
        print(f"\nDataset Summary:")
        print(f"  Training:   {len(self.train_dataset)} samples")
        print(f"  Validation: {len(self.val_dataset)} samples")
        print(f"  Testing:    {len(self.test_dataset)} samples")
        print(f"  EEG shape:  {self.Chans} channels × {self.Samples} time points")
        
        # 初始化EEGNet模型
        print(f"\nInitializing EEGNet model...")
        self.model = EEGFormer(eeg_channel=30)
        
        print(f"\nModel Configuration:")
        print(f"  Model type: EEGNet")
        print(f"  Number of classes: {num_classes}")
        print(f"  Channels: {self.Chans}")
        print(f"  Samples: {self.Samples}")
        print(f"  Dropout rate: {dropoutRate}")
        print(f"  F1={F1}, D={D}, F2={F2}")
        print(f"  Device: {self.device}")
        
        # 移动模型到设备
        self.model.to(self.device)
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器将在train()方法中初始化
        self.scheduler = None
        
        # 创建dataloaders
        print("\nPreparing dataloaders...")
        self._create_dataloaders()
        
        print("="*80)
        print("Initialization complete!")
        print("="*80)
        sys.stdout.flush()
    
    def _create_dataloaders(self):
        """创建dataloaders"""
        print(f"Creating dataloaders with batch_size={self.batch_size}...")
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"✓ Dataloaders ready!")
        print(f"  Train batches: {len(self.train_dataloader)}")
        print(f"  Val batches:   {len(self.val_dataloader)}")
        print(f"  Test batches:  {len(self.test_dataloader)}")
    
    def train_epoch(self, epoch, total_epochs):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        with tqdm(self.train_dataloader,
                 desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                 unit="batch", file=sys.stdout, dynamic_ncols=True) as pbar:
            
            for batch_idx, batch in enumerate(pbar, start=1):
                # 提取EEG数据和标签
                eeg_data = batch['eeg']  # (B, Chans, Samples)
                labels = batch['label']
                
                # 移到GPU
                eeg_data = eeg_data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(eeg_data)
                
                # 计算损失（使用CrossEntropyLoss）
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 获取当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{epoch_loss/batch_idx:.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'lr': f'{current_lr:.6f}'
                })
        
        avg_loss = epoch_loss / len(self.train_dataloader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def evaluate(self, dataloader, split_name='Val'):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            with tqdm(dataloader,
                     desc=f"[{split_name}]",
                     unit="batch", file=sys.stdout, dynamic_ncols=True) as pbar:
                
                for batch_idx, batch in enumerate(pbar, start=1):
                    # 提取数据
                    eeg_data = batch['eeg']
                    labels = batch['label']
                    
                    eeg_data = eeg_data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    # 前向传播
                    outputs = self.model(eeg_data)
                    loss = self.criterion(outputs, labels)
                    
                    # 统计
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 收集预测结果
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_outputs.append(outputs.cpu().numpy())
                    
                    # 更新进度条
                    avg_acc = 100. * correct / total
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{avg_acc:.2f}%'
                    })
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100. * correct / total
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_outputs = np.concatenate(all_outputs, axis=0)
        
        return avg_loss, avg_acc, all_predictions, all_labels, all_outputs
    
    def save_checkpoint(self, epoch, train_loss, val_loss, val_acc):
        """保存最佳模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'model_config': {
                'num_classes': self.num_classes,
                'Chans': self.Chans,
                'Samples': self.Samples
            }
        }
        
        # 删除之前保存的最佳模型
        best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
        if os.path.exists(best_path):
            try:
                os.remove(best_path)
            except Exception as e:
                print(f"  Warning: Failed to remove old checkpoint: {e}")
        
        # 保存新的最佳模型
        torch.save(checkpoint, best_path)
        print(f"  ✓ Saved best checkpoint (epoch {epoch}, val_acc: {val_acc:.2f}%)")
        
        return best_path
    
    def load_checkpoint(self, checkpoint_path):
        """加载checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型和优化器状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载scheduler状态（如果存在）
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Acc: {checkpoint['val_acc']:.2f}%")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint
    
    def train(self, epochs=100, early_stopping_patience=10, use_validation=True,
              resume_from=None, scheduler_type='cosine', T_max=None, eta_min=1e-6):
        """训练模型"""
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {self.lr}")
        print(f"  LR Scheduler: {scheduler_type}")
        if scheduler_type == 'cosine':
            print(f"  T_max: {T_max or epochs}")
            print(f"  eta_min: {eta_min}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"  Using validation: {use_validation}")
        if resume_from:
            print(f"  Resume from: {resume_from}")
        print(f"{'='*80}\n")
        sys.stdout.flush()
        
        # 初始化学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=T_max or epochs,
                eta_min=eta_min
            )
            print(f"Using CosineAnnealingLR scheduler (T_max={T_max or epochs}, eta_min={eta_min})")
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
            print(f"Using StepLR scheduler")
        elif scheduler_type == 'none':
            self.scheduler = None
            print(f"No LR scheduler")
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # 如果需要恢复训练
        start_epoch = 0
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            if checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resuming training from epoch {start_epoch}")
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        train_history = []
        val_history = []
        
        for epoch in range(start_epoch, epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch, epochs)
            
            # 验证
            if use_validation:
                val_loss, val_acc, _, _, _ = self.evaluate(self.val_dataloader, 'Val')
            else:
                val_loss, val_acc, _, _, _ = self.evaluate(self.test_dataloader, 'Test')
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.lr
            
            # 记录历史
            train_history.append({
                'loss': train_loss, 
                'acc': train_acc, 
                'lr': current_lr
            })
            val_history.append({
                'loss': val_loss, 
                'acc': val_acc
            })
            
            # 打印摘要
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  {'Val' if use_validation else 'Test'} - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # 检查是否是最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                print(f"  ✓ New best {'validation' if use_validation else 'test'} accuracy!")
                
                # 只在最佳时保存checkpoint
                self.save_checkpoint(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_acc=val_acc
                )
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} epoch(s)")
            
            print(f"{'-'*80}")
            sys.stdout.flush()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
                print(f"{'='*80}\n")
                break
        
        # 加载最佳模型
        best_ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
        if os.path.exists(best_ckpt_path):
            print(f"\nLoading best model from {best_ckpt_path}...")
            self.load_checkpoint(best_ckpt_path)
        
        print(f"\nTraining completed!")
        print(f"Best {'validation' if use_validation else 'test'} accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
        
        return train_history, val_history
    
    def final_evaluate(self):
        """在测试集上进行最终评估"""
        print(f"\n{'='*80}")
        print(f"Final Evaluation on Test Set")
        print(f"{'='*80}")
        
        test_loss, test_acc, predictions, labels, outputs = self.evaluate(
            self.test_dataloader, 'Test'
        )
        
        # 计算更多指标
        from sklearn.metrics import f1_score, classification_report, confusion_matrix
        
        f1 = f1_score(labels, predictions, average='weighted')
        f1_per_class = f1_score(labels, predictions, average=None)
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.2f}%")
        print(f"  F1-Score (weighted): {f1:.4f}")
        print(f"\nF1-Score per class:")
        for i, f1_score_class in enumerate(f1_per_class):
            print(f"  Class {i}: {f1_score_class:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(labels, predictions))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(labels, predictions)
        print(cm)
        
        print(f"{'='*80}\n")
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_f1': f1,
            'predictions': predictions,
            'labels': labels,
            'outputs': outputs,
            'confusion_matrix': cm,
            'f1_per_class': f1_per_class
        }


def main():
    """主函数"""
    print("\n" + "="*80)
    print("Cross-Subject EEGNet Training for Emotion Recognition")
    print("="*80)
    
    # 配置参数
    data_root_dict = {
        'audio': '/data2/mingzhi/BCI/dataset/EAV/Audio',
        'eeg': '/data2/mingzhi/BCI/dataset/EAV/EEG',
        'vision': '/data2/mingzhi/BCI/dataset/EAV/Vision'
    }
    
    csv_path = "/data2/mingzhi/BCI/dataset/EAV/subjects.csv"
    
    try:
        # 初始化训练器
        trainer = CrossSubjectEEGNetTrainer(
            data_root_dict=data_root_dict,
            num_classes=5,
            dropoutRate=0.5,
            kernLength=64,
            F1=8,
            D=2,
            F2=16,
            lr=0.001,
            batch_size=64,
            num_workers=4,
            random_seed=2024,
            subject_csv_path=csv_path if os.path.exists(csv_path) else None
        )
        
        # 训练模型
        train_history, val_history = trainer.train(
            epochs=100,
            early_stopping_patience=30,
            use_validation=True,
            resume_from=None,
            scheduler_type='cosine',
            T_max=100,
            eta_min=1e-6
        )
        
        # 最终评估
        results = trainer.final_evaluate()
        
        # 保存结果
        import pickle
        
        full_results = {
            'test_results': results,
            'train_history': train_history,
            'val_history': val_history,
            'train_subjects': trainer.dataset_loader.train_subjects,
            'val_subjects': trainer.dataset_loader.val_subjects,
            'test_subjects': trainer.dataset_loader.test_subjects
        }
        
        with open('eegnet_cross_subject_results.pkl', 'wb') as f:
            pickle.dump(full_results, f)
        
        print(f"✓ Results saved to eegnet_cross_subject_results.pkl")
        print("="*80)
        
        # 打印最终结果摘要
        print("\nFinal Summary:")
        print(f"  Test Accuracy: {results['test_accuracy']:.2f}%")
        print(f"  Test F1-Score: {results['test_f1']:.4f}")
        print(f"  Train subjects: {len(trainer.dataset_loader.train_subjects)}")
        print(f"  Val subjects: {len(trainer.dataset_loader.val_subjects)}")
        print(f"  Test subjects: {len(trainer.dataset_loader.test_subjects)}")
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理内存
        if 'trainer' in locals():
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()