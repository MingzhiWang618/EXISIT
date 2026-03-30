import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')
import time
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import ASTFeatureExtractor, ASTModel
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from dataset.dataset import CrossSubjectMultiModalDataset, MultiModalTensorDataset

# -----------------------------
# 工具函数
# -----------------------------
def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


# -----------------------------
# AST Audio Processor
# -----------------------------
class ASTAudioProcessor:
    """
    使用 HuggingFace AST 处理 waveform numpy array
    """
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", sample_rate=16000):
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.sample_rate = sample_rate
        
    def preprocess_batch(self, audio_batch, device='cuda'):
        """
        audio_batch: list or tensor of shape [batch, waveform_length] (float32)
        返回: dict with 'input_values' of shape [batch, 1, time, freq]
        """
        waveforms = []
        for waveform in audio_batch:
            if torch.is_tensor(waveform):
                waveform = waveform.cpu().numpy()
            waveforms.append(waveform)
        
        # 使用feature extractor处理
        inputs = self.feature_extractor(
            waveforms, 
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


# -----------------------------
# AST音频分类器
# -----------------------------
class ASTAudioClassifier(nn.Module):
    def __init__(self, num_labels=5, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", 
                 hidden_dim=768):
        super().__init__()
        
        # 加载预训练的AST模型
        self.ast = ASTModel.from_pretrained(model_name)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, input_values):
        """
        input_values: [batch, 1, time, freq] from feature extractor
        """
        # AST forward pass
        outputs = self.ast(input_values)
        
        # 使用 [CLS] token 的输出 (last_hidden_state[:, 0])
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # 分类
        logits = self.classifier(pooled_output)
        return logits


# -----------------------------
# 训练器 (Two-Phase Training)
# -----------------------------
class CrossSubjectASTTrainer:
    def __init__(self, data_root_dict, model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 num_labels=5, batch_size=16, unfrozen_batch_size=8, num_workers=4, 
                 lr=5e-5, random_seed=2024, checkpoint_dir="./ast_checkpoints",
                 subject_csv_path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.unfrozen_batch_size = unfrozen_batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # -----------------------------
        # 数据集
        # -----------------------------
        self.dataset_loader = CrossSubjectMultiModalDataset(
            data_root_dict=data_root_dict,
            random_seed=random_seed,
            subject_csv_path=subject_csv_path
        )
        train_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('train'))
        val_data   = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('val'))
        test_data  = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('test'))
        self.datasets = {'train': train_data, 'val': val_data, 'test': test_data}

        # -----------------------------
        # 初始化 AST Processor & Model
        # -----------------------------
        print("Initializing AST processor and model...")
        self.processor = ASTAudioProcessor(model_name=model_name)
        self.model = ASTAudioClassifier(num_labels=num_labels, model_name=model_name).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr, 
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=20, 
            eta_min=1e-6
        )

        # DataLoader (初始使用batch_size)
        self._create_dataloaders(self.batch_size)

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def _create_dataloaders(self, batch_size):
        self.dataloaders = {
            split: DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'),
                              num_workers=self.num_workers, pin_memory=True)
            for split, ds in self.datasets.items()
        }

    def _forward_batch(self, batch, is_train=False):
        """
        Forward a batch.
        - if is_train=True: returns (loss_tensor, acc_float) with loss requiring grad
        - if is_train=False: returns (loss_float, acc_float) computed under no_grad()
        """
        audio_waveforms, labels = batch['audio'], batch['label']
        inputs = self.processor.preprocess_batch(audio_waveforms, device=self.device)
        labels = labels.to(self.device, non_blocking=True)
        
        if is_train:
            logits = self.model(inputs['input_values'])
            loss = self.criterion(logits, labels)
            acc = calculate_accuracy(logits, labels)
            return loss, float(acc)
        else:
            with torch.no_grad():
                logits = self.model(inputs['input_values'])
                loss = self.criterion(logits, labels)
                acc = calculate_accuracy(logits, labels)
            return float(loss.item()), float(acc)

    def _run_epoch(self, split='train'):
        is_train = (split == 'train')
        dataloader = self.dataloaders[split]

        if is_train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        with tqdm(dataloader, file=sys.stdout, dynamic_ncols=True) as pbar:
            for batch_idx, batch in enumerate(pbar, start=1):
                try:
                    loss_ret, acc_val = self._forward_batch(batch, is_train=is_train)
                    
                    if is_train:
                        if not isinstance(loss_ret, torch.Tensor):
                            raise RuntimeError("Expected loss tensor in training mode")
                        loss_ret.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        loss_val = float(loss_ret.item())
                    else:
                        loss_val = float(loss_ret)

                    total_loss += loss_val
                    total_acc += acc_val
                    steps += 1

                    pbar.set_postfix({
                        'loss': f'{(total_loss/steps):.4f}',
                        'acc': f'{(total_acc/steps)*100:.2f}%'
                    })

                    # 定期清理显存
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    del batch, loss_ret, acc_val

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("WARNING: OOM during epoch; clearing cache.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if is_train:
                            self.optimizer.zero_grad()
                        continue
                    else:
                        raise e

        avg_loss = (total_loss / steps) if steps > 0 else 0.0
        avg_acc = (total_acc / steps) if steps > 0 else 0.0
        return avg_loss, avg_acc

    def train(self, epochs=3, freeze=True, use_validation=True, save_every_epoch=True):
        """
        两阶段训练：
        - freeze=True: 冻结AST backbone，只训练分类头
        - freeze=False: 端到端fine-tuning
        """
        # 冻结/解冻 backbone
        for param in self.model.ast.parameters():
            param.requires_grad = not freeze
        
        # 确保分类头始终可训练
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # 根据freeze状态调整batch size
        batch_size = self.unfrozen_batch_size if not freeze else self.batch_size
        self._create_dataloaders(batch_size)

        # 重新创建optimizer（只包含需要训练的参数）
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs, 
            eta_min=1e-6
        )

        best_val_acc = -1.0
        for epoch in range(1, epochs + 1):
            start_ts = time.time()
            print(f"\n=== Epoch {epoch}/{epochs} | freeze={freeze} | batch_size={batch_size} ===")

            train_loss, train_acc = self._run_epoch('train')
            print(f"Train   - loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

            val_loss, val_acc = (0.0, 0.0)
            if use_validation:
                val_loss, val_acc = self._run_epoch('val')
                print(f"Validate - loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%")

            # Scheduler step
            self.scheduler.step()
            print(f"[Scheduler] lr now = {self.scheduler.get_last_lr()[0]:.6f}")

            # Save checkpoint
            if save_every_epoch:
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'freeze': freeze
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            # Save best model
            if use_validation and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc
                }, best_path)
                print(f"New best model saved to {best_path} (val_acc={val_acc*100:.2f}%)")

            print(f"Epoch {epoch} finished in {time.time()-start_ts:.1f}s")

    def evaluate(self, split='test'):
        avg_loss, avg_acc = self._run_epoch(split)
        print(f"{split.capitalize()} - loss: {avg_loss:.4f}, acc: {avg_acc*100:.2f}%")

        # 收集预测结果用于详细评估
        dataloader = self.dataloaders[split]
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}", file=sys.stdout):
                audio_waveforms, labels = batch['audio'], batch['label']
                inputs = self.processor.preprocess_batch(audio_waveforms, device=self.device)
                labels = labels.to(self.device, non_blocking=True)
                
                logits = self.model(inputs['input_values'])
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 计算F1和详细报告
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        
        return avg_acc, f1




    def load_checkpoint(self, checkpoint_path):
        """加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        if 'val_acc' in checkpoint:
            print(f"Checkpoint validation accuracy: {checkpoint['val_acc']*100:.2f}%")


# -----------------------------
# Main
# -----------------------------
def main():
    print("\n" + "=" * 80)
    print("Cross-Subject AST Training (Two-Phase)")
    print("=" * 80)

    data_root_dict = {
        'audio': '/data2/mingzhi/BCI/dataset/EAV_old/Audio',
        'eeg': '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision'
    }

    # CSV路径
    csv_path = "/data2/mingzhi/BCI/dataset/EAV_old/subjects.csv"
    
    try:
        trainer = CrossSubjectASTTrainer(
            data_root_dict=data_root_dict,
            model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=5,
            batch_size=32,           # Phase 1 (frozen) batch size
            unfrozen_batch_size=32,   # Phase 2 (unfrozen) batch size
            num_workers=8,
            lr=1e-3,
            checkpoint_dir="./ast_checkpoints",
            subject_csv_path=csv_path if os.path.exists(csv_path) else None
        )

        # Phase 1: 冻结backbone训练
        print("\n" + "=" * 80)
        print("PHASE 1: Frozen Backbone Training (Classifier Only)")
        print("=" * 80)
        trainer.train(epochs=30, freeze=True, use_validation=True, save_every_epoch=True)

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 2: 端到端fine-tuning
        print("\n" + "=" * 80)
        print("PHASE 2: Unfrozen Fine-tuning (End-to-End)")
        print("=" * 80)
        trainer.train(epochs=0, freeze=False, use_validation=True, save_every_epoch=True)

        # 最终评估
        print("\n" + "=" * 80)
        print("FINAL EVALUATION on Test Set")
        print("=" * 80)
        test_acc, test_f1 = trainer.evaluate(split='test')
        print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
        print(f"Final Test F1 Score: {test_f1:.4f}")

        # 保存最终结果
        results = {
            'test_acc': test_acc,
            'test_f1': test_f1,
            'train_subjects': trainer.dataset_loader.train_subjects,
            'val_subjects': trainer.dataset_loader.val_subjects,
            'test_subjects': trainer.dataset_loader.test_subjects,
        }
        with open("ast_training_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("\nResults saved to ast_training_results.pkl")

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
