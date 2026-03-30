#!/usr/bin/env python3
# VideoMAE_full_trainer_ddp_hf_gpu_preproc.py
# Multi-GPU (single-node) version using torch.distributed (DDP)
# Uses HuggingFace pre-trained VideoMAE automatically
# GPU-native preprocessing to avoid PIL/numpy bottleneck
# Launch with: torchrun --nproc_per_node=NUM_GPUS VideoMAE_full_trainer_ddp_hf_gpu_preproc.py

import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

import time
import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F

from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ---------------- Distributed utils -----------------
def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def setup_distributed():
    """Initialize DDP process group if torchrun is used."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)
        return True
    return False


# ---------------- Accuracy -----------------
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.logits, dim=1)
    return (predicted == labels).sum().item() / labels.size(0)


# ---------------- GPU Video Preprocessing -----------------
class VideoMAEVideoPreprocessor:
    """
    GPU-native video preprocessor for VideoMAE:
      - resize frames to (224,224)
      - linear temporal interpolation to target_frames
      - normalize using processor's mean/std
    """

    def __init__(self, processor, target_frames: int = 16, device='cuda'):
        self.processor = processor
        self.target_frames = target_frames
        self.device = device

        self.mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _temporal_interpolate(self, video_tensor: torch.Tensor):
        """
        video_tensor: (T, C, H, W)
        returns: (target_frames, C, H, W)
        """
        T, C, H, W = video_tensor.shape
        if T == self.target_frames:
            return video_tensor
        video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
        video_tensor = F.interpolate(video_tensor, size=(self.target_frames, H, W), mode='trilinear', align_corners=False)
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).squeeze(0)  # (target_frames, C, H, W)
        return video_tensor

    def preprocess_batch(self, video_batch):
        """
        video_batch: iterable of videos (each video: torch.Tensor of shape (T,C,H,W) or (T,H,W,C))
        returns: tensor (B, T, C, H, W) on self.device
        """
        processed_videos = []
        for video in video_batch:
            if not isinstance(video, torch.Tensor):
                video = torch.tensor(video, dtype=torch.float32)
            video = video.to(self.device, non_blocking=True)

            # if video is (T,H,W,C) -> (T,C,H,W)
            if video.ndim == 4 and video.shape[3] in (1, 3):
                video = video.permute(0, 3, 1, 2).contiguous()

            # normalize to [0,1] if needed
            if video.max() > 1.0:
                video = video / 255.0

            # resize frames to 224x224 using bilinear
            video = F.interpolate(video.unsqueeze(0), size=(video.shape[1], 224, 224), mode='trilinear', align_corners=False)
            video = video.squeeze(0)  # (T, C, H, W)

            # temporal interpolation
            video = self._temporal_interpolate(video)

            # normalize
            video = (video - self.mean) / self.std
            processed_videos.append(video)

        return torch.stack(processed_videos, dim=0)  # (B, T, C, H, W)


# ---------------- Trainer -----------------
class CrossSubjectVideoMAETrainer:
    """Cross-subject VideoMAE trainer with optional DDP support."""

    def __init__(self, data_root_dict, model_name_or_path,
                 num_labels=5, batch_size=8, unfrozen_batch_size=1,
                 num_workers=4, random_seed: int = 2024,
                 subject_csv_path: Optional[str] = None,
                 lr: float = 5e-5,
                 checkpoint_dir: str = "./checkpoints",
                 distributed: bool = False):
        self.distributed = distributed
        self.world_size = get_world_size()
        self.rank = get_rank()
        if self.distributed and "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.unfrozen_batch_size = unfrozen_batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        np.random.seed(random_seed + self.rank)
        torch.manual_seed(random_seed + self.rank)

        # ---------------- Dataset -----------------
        from dataset.dataset import CrossSubjectMultiModalDataset, MultiModalTensorDataset
        self.dataset_loader = CrossSubjectMultiModalDataset(
            data_root_dict=data_root_dict,
            random_seed=random_seed,
            subject_csv_path=subject_csv_path
        )
        train_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('train'))
        val_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('val'))
        test_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('test'))
        self.datasets = {'train': train_data, 'val': val_data, 'test': test_data}

        # ---------------- Model & Processor -----------------
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name_or_path)
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name_or_path,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if torch.cuda.is_available() else None,
                             find_unused_parameters=True)

        self.target_frames = getattr(self.model.module.config if hasattr(self.model, "module") else self.model.config, 'num_frames', 16)
        self.video_preprocessor = VideoMAEVideoPreprocessor(self.processor, self.target_frames, device=self.device)

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-6)
        self.criterion = nn.CrossEntropyLoss()

        self._create_dataloaders(self.batch_size)

        if self.rank == 0:
            print(f"Initialized trainer on {self.device}, num_labels={self.num_labels}, target_frames={self.target_frames}, distributed={self.distributed}, world_size={self.world_size}")


    def _create_dataloaders(self, batch_size):
        samplers = {}
        for split, ds in self.datasets.items():
            if self.distributed:
                samplers[split] = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank,
                                                     shuffle=(split=='train'))
            else:
                samplers[split] = None

        self.dataloaders = {}
        for split, ds in self.datasets.items():
            sampler = samplers[split]
            shuffle = (split == 'train' and sampler is None)
            workers = max(0, int(self.num_workers / max(1, self.world_size)))
            self.dataloaders[split] = DataLoader(ds,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 sampler=sampler,
                                                 num_workers=workers,
                                                 pin_memory=True)

    # ---------------- Forward / Epoch -----------------
    def _forward_batch(self, batch, is_train: bool = False):
        video_data, labels = batch['vision'], batch['label']
        pixel_values = self.video_preprocessor.preprocess_batch(video_data).to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if is_train:
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            if getattr(loss, "dim", lambda: 0)() > 0:
                loss = loss.mean()
            acc = calculate_accuracy(outputs, labels)
            return loss, float(acc)
        else:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                loss_tensor = self.criterion(logits, labels)
                loss_value = float(loss_tensor.item())
                acc = calculate_accuracy(outputs, labels)
            return loss_value, float(acc)

    def _run_epoch(self, split='train'):
        is_train = (split == 'train')
        dataloader = self.dataloaders[split]
        sampler = dataloader.sampler if hasattr(dataloader, "sampler") else None

        if is_train:
            self.model.train()
            if sampler is not None and isinstance(sampler, DistributedSampler):
                sampler.set_epoch(int(time.time()) % 100000)
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        pbar = tqdm(dataloader, file=sys.stdout, dynamic_ncols=True) if self.rank == 0 else dataloader
        for batch_idx, batch in enumerate(pbar, start=1):
            try:
                loss_ret, acc_val = self._forward_batch(batch, is_train=is_train)

                if is_train:
                    loss_ret.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    loss_val = float(loss_ret.item())
                else:
                    loss_val = float(loss_ret)

                total_loss += loss_val
                total_acc += acc_val
                steps += 1

                if self.rank == 0:
                    pbar.set_postfix({'loss': f'{(total_loss/steps):.4f}', 'acc': f'{(total_acc/steps)*100:.2f}%'})

                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                del batch, loss_ret, acc_val

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.rank == 0:
                        print("WARNING: OOM; clearing cache and continuing")
                    torch.cuda.empty_cache()
                    if is_train:
                        self.optimizer.zero_grad()
                    continue
                else:
                    raise e

        avg_loss = (total_loss / steps) if steps > 0 else 0.0
        avg_acc = (total_acc / steps) if steps > 0 else 0.0

        if self.distributed:
            tb = torch.tensor([avg_loss, avg_acc], dtype=torch.float64, device=self.device)
            dist.all_reduce(tb, op=dist.ReduceOp.SUM)
            tb = tb / float(self.world_size)
            avg_loss = float(tb[0].item())
            avg_acc = float(tb[1].item())

        return avg_loss, avg_acc

    # ---------------- Train / Evaluate -----------------
    def train(self, epochs=3, freeze=True, use_validation=True, save_every_epoch: bool = True):
        backbone = getattr(self.model.module if hasattr(self.model, "module") else self.model, 'videomae', self.model)
        for p in backbone.parameters():
            p.requires_grad = not freeze
        if hasattr(self.model, 'classifier') or (hasattr(self.model, "module") and hasattr(self.model.module, "classifier")):
            classifier = self.model.module.classifier if hasattr(self.model, "module") else self.model.classifier
            for p in classifier.parameters():
                p.requires_grad = True

        batch_size = self.unfrozen_batch_size if not freeze else self.batch_size
        self._create_dataloaders(batch_size)

        best_val_acc = -1.0
        for epoch in range(1, epochs + 1):
            start_ts = time.time()
            if self.rank == 0:
                print(f"\n=== Epoch {epoch}/{epochs} | freeze={freeze} | batch_size_per_process={batch_size} ===")
            train_loss, train_acc = self._run_epoch('train')
            if self.rank == 0:
                print(f"Train  - loss: {train_loss:.4f}, acc: {train_acc*100:.2f}%")

            val_loss, val_acc = (0.0, 0.0)
            if use_validation:
                val_loss, val_acc = self._run_epoch('val')
                if self.rank == 0:
                    print(f"Validate - loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%")

            if save_every_epoch and self.rank == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"best_checkpoint_epoch_{epoch}.pt")
                model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            self.scheduler.step()
            if self.rank == 0:
                print(f"[Scheduler] lr now = {self.scheduler.get_last_lr()[0]:.6f}")

            elapsed = time.time() - start_ts
            if self.rank == 0:
                print(f"Epoch {epoch} finished in {elapsed:.1f}s")

    def evaluate(self, split='test'):
        avg_loss, avg_acc = self._run_epoch(split)
        if self.rank == 0:
            print(f"{split.capitalize()} - loss: {avg_loss:.4f}, acc: {avg_acc*100:.2f}%")
        return avg_acc


# ---------------- Main -----------------
def main():
    distributed = setup_distributed()
    rank = get_rank()

    if rank == 0:
        print("\n" + "="*80)
        print("Cross-Subject Video-MAE Training (DDP-enabled, HuggingFace pre-trained, GPU-native preprocessing)")
        print("="*80)

    data_root_dict = {
        'audio': '/data2/mingzhi/BCI/dataset/EAV_old/Audio',
        'eeg': '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision'
    }

    model_name = "MCG-NJU/videomae-base"
    csv_path = "/data2/mingzhi/BCI/dataset/EAV_old/subjects.csv"

    trainer = CrossSubjectVideoMAETrainer(
        data_root_dict=data_root_dict,
        model_name_or_path=model_name,
        num_labels=5,
        batch_size=32,
        unfrozen_batch_size=4,
        num_workers=32,
        random_seed=2024,
        subject_csv_path=csv_path if os.path.exists(csv_path) else None,
        lr=5e-5,
        checkpoint_dir="./checkpoints",
        distributed=distributed
    )

    if rank == 0:
        print("\nPHASE 1: Frozen Feature Training")
    trainer.train(epochs=30, freeze=True, use_validation=True, save_every_epoch=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rank == 0:
        print("\nPHASE 2: Unfrozen Fine-tuning")
    trainer.train(epochs=0, freeze=False, use_validation=True, save_every_epoch=True)

    if rank == 0:
        print("\nFINAL EVALUATION on test set")
    acc = trainer.evaluate(split='test')
    if rank == 0:
        print(f"Final Test Acc: {acc*100:.2f}%")

        results = {'test_acc': acc,
                   'train_subjects': getattr(trainer.dataset_loader, 'train_subjects', None),
                   'val_subjects': getattr(trainer.dataset_loader, 'val_subjects', None),
                   'test_subjects': getattr(trainer.dataset_loader, 'test_subjects', None)}
        with open('cross_subject_videomae_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("Saved results to cross_subject_videomae_results.pkl")


if __name__ == "__main__":
    main()