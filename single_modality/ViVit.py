#!/usr/bin/env python3
# ViVit_full_trainer_ddp.py
# Multi-GPU (single-node) version using torch.distributed (DDP).
# Launch with: torchrun --nproc_per_node=NUM_GPUS ViVit_full_trainer_ddp.py

import os
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI/archieve')

import time
import pickle
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

# Distributed utils
import torch.distributed as dist

from transformers import VivitForVideoClassification, VivitImageProcessor
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def setup_distributed():
    """
    Initialize distributed process group if environment variables indicate distributed launch.
    torchrun / torch.distributed.launch sets:
      - RANK
      - LOCAL_RANK
      - WORLD_SIZE
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        torch.cuda.set_device(local_rank)
        return True
    return False


def calculate_accuracy(outputs, labels):
    """Compute batch accuracy from model outputs (assumes outputs.logits)."""
    _, predicted = torch.max(outputs.logits, dim=1)
    return (predicted == labels).sum().item() / labels.size(0)


class ViViTVideoPreprocessor:
    """Video preprocessor using temporal linear interpolation to target_frames."""

    def __init__(self, processor, target_frames: int = 25):
        self.processor = processor
        self.target_frames = target_frames

    def _adjust_frames(self, frames_list):
        n = len(frames_list)
        if n == 0:
            raise ValueError("frames_list is empty")
        if n == 1:
            return [frames_list[0]] * self.target_frames

        arrs = [np.asarray(im).astype(np.float32) for im in frames_list]
        arrs = np.stack(arrs, axis=0)  # (n, H, W, C)

        sample_positions = np.linspace(0, n - 1, self.target_frames)
        out_frames = []
        for pos in sample_positions:
            left = int(np.floor(pos))
            right = int(np.ceil(pos))
            if left == right:
                frame_arr = arrs[left]
            else:
                w = pos - left
                frame_arr = (1.0 - w) * arrs[left] + w * arrs[right]
            frame_arr = np.clip(frame_arr, 0, 255).astype(np.uint8)
            out_frames.append(Image.fromarray(frame_arr))
        return out_frames

    def preprocess_batch(self, video_batch):
        """
        video_batch: iterable of videos.
        Each video: torch.Tensor (T,C,H,W) or (T,H,W,C) or numpy array
        Returns: tensor (B, T, C, H, W)
        """
        processed_videos = []
        for video in video_batch:
            if isinstance(video, torch.Tensor):
                video = video.cpu().numpy()
            # handle (T,C,H,W) -> (T,H,W,C)
            if video.ndim == 4 and video.shape[1] in (1, 3):
                video = np.transpose(video, (0, 2, 3, 1))
            frames = []
            for frame in video:
                if frame.dtype != np.uint8:
                    # assume normalized [0,1] or larger range
                    if frame.max() <= 1.0:
                        frame = (frame * 255.0).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                # ensure HWC
                frame_pil = Image.fromarray(frame).resize((224, 224), Image.BILINEAR)
                frames.append(frame_pil)
            frames = self._adjust_frames(frames)
            processed = self.processor(frames, return_tensors="pt")
            processed_videos.append(processed.pixel_values.squeeze(0))
        return torch.stack(processed_videos)  # (B, T, C, H, W)


class CrossSubjectViViTTrainer:
    """
    Clean Cross-Subject ViViT trainer adapted to DDP:
      - training returns loss tensor (no duplicate forward)
      - validation computes loss (no backward) and accuracy
      - checkpointing only on rank 0
      - uses DistributedSampler when distributed
    """

    def __init__(self, data_root_dict, model_path,
                 num_labels=5, batch_size=8, unfrozen_batch_size=1,
                 num_workers=4, random_seed: int = 2024,
                 subject_csv_path: Optional[str] = None,
                 lr: float = 5e-5,
                 checkpoint_dir: str = "./checkpoints",
                 distributed: bool = False):
        self.distributed = distributed
        self.world_size = get_world_size()
        self.rank = get_rank()
        # device: if distributed, set per-process GPU, else auto
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

        # Safe seed conversion
        try:
            seed_int = int(str(random_seed).strip())
        except Exception:
            seed_int = 2024
            if self.rank == 0:
                print(f"WARNING: invalid random_seed {random_seed!r}, using {seed_int}")
        np.random.seed(seed_int + self.rank)
        torch.manual_seed(seed_int + self.rank)

        # Load dataset (adjust import path for your project)
        from dataset.dataset import CrossSubjectMultiModalDataset, MultiModalTensorDataset
        self.dataset_loader = CrossSubjectMultiModalDataset(
            data_root_dict=data_root_dict,
            random_seed=seed_int,
            subject_csv_path=subject_csv_path
        )
        train_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('train'))
        val_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('val'))
        test_data = MultiModalTensorDataset(self.dataset_loader.load_all_modalities_by_split('test'))
        self.datasets = {'train': train_data, 'val': val_data, 'test': test_data}

        # Processor & model
        # IMPORTANT: instantiate model on correct device before wrapping with DDP
        self.processor = VivitImageProcessor.from_pretrained(model_path)
        self.model = VivitForVideoClassification.from_pretrained(
            model_path, num_labels=self.num_labels, ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

        # Wrap with DDP if distributed
        if self.distributed:
            # Delay import of DDP to avoid errors on CPU-only setups
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if torch.cuda.is_available() else None,
            find_unused_parameters=True)

        self.target_frames = getattr(self.model.module.config if hasattr(self.model, "module") else self.model.config, 'num_frames', 32)
        self.video_preprocessor = ViViTVideoPreprocessor(self.processor, self.target_frames)
        # optimizer should only contain parameters that require grad
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20,
            eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()

        # dataloaders
        self._create_dataloaders(self.batch_size)

        if self.rank == 0:
            print(f"Initialized trainer on {self.device}, num_labels={self.num_labels}, target_frames={self.target_frames}, distributed={self.distributed}, world_size={self.world_size}")

    def _create_dataloaders(self, batch_size):
        # If distributed, adjust batch_size to be per-process batch size passed in already.
        samplers = {}
        for split, ds in self.datasets.items():
            if self.distributed and split == 'train':
                samplers[split] = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            elif self.distributed:
                # for val/test use sequential distributed sampler to split evaluation across ranks
                samplers[split] = DistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            else:
                samplers[split] = None

        self.dataloaders = {}
        for split, ds in self.datasets.items():
            sampler = samplers[split]
            shuffle = (split == 'train' and sampler is None)
            # num_workers can be scaled down per process to avoid too many threads
            workers = max(0, int(self.num_workers / max(1, self.world_size)))
            self.dataloaders[split] = DataLoader(ds,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 sampler=sampler,
                                                 num_workers=workers,
                                                 pin_memory=True)

    def _forward_batch(self, batch, is_train: bool = False):
        """
        Forward a batch.
        - if is_train=True: returns (loss_tensor, acc_float) with loss tensor requiring grad
        - if is_train=False: returns (loss_float, acc_float) computed under no_grad()
        """
        video_data, labels = batch['vision'], batch['label']
        pixel_values = self.video_preprocessor.preprocess_batch(video_data).to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if is_train:
            outputs = self.model(pixel_values=pixel_values, labels=labels, interpolate_pos_encoding=True)
            loss = outputs.loss
            if getattr(loss, "dim", lambda: 0)() > 0:
                loss = loss.mean()
            acc = calculate_accuracy(outputs, labels)
            return loss, float(acc)
        else:
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values, interpolate_pos_encoding=True)
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

        # progress bar only on rank 0 to avoid mixed output
        pbar = tqdm(dataloader, file=sys.stdout, dynamic_ncols=True) if self.rank == 0 else dataloader
        for batch_idx, batch in enumerate(pbar, start=1):
            try:
                loss_ret, acc_val = self._forward_batch(batch, is_train=is_train)

                if is_train:
                    if not isinstance(loss_ret, torch.Tensor):
                        raise RuntimeError("Expected loss tensor in training mode")
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
                    pbar.set_postfix({
                        'loss': f'{(total_loss/steps):.4f}',
                        'acc': f'{(total_acc/steps)*100:.2f}%'
                    })

                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # explicit cleanup
                del batch, loss_ret, acc_val

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.rank == 0:
                        print("WARNING: OOM during epoch; clearing cache and continuing.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if is_train:
                        self.optimizer.zero_grad()
                    continue
                else:
                    raise e

        # aggregate metrics across processes if distributed
        avg_loss = (total_loss / steps) if steps > 0 else 0.0
        avg_acc = (total_acc / steps) if steps > 0 else 0.0

        if self.distributed:
            # reduce avg_loss and avg_acc across processes (mean)
            tb = torch.tensor([avg_loss, avg_acc], dtype=torch.float64, device=self.device)
            dist.all_reduce(tb, op=dist.ReduceOp.SUM)
            tb = tb / float(self.world_size)
            avg_loss = float(tb[0].item())
            avg_acc = float(tb[1].item())

        return avg_loss, avg_acc

    def train(self, epochs=3, freeze=True, use_validation=True, save_every_epoch: bool = True):
        # freeze/unfreeze backbone
        backbone = getattr(self.model.module if hasattr(self.model, "module") else self.model, 'vivit', self.model)
        # When freeze=True we want to freeze backbone; classifier remains trainable
        for p in backbone.parameters():
            p.requires_grad = not freeze
        if hasattr(self.model, 'classifier') or (hasattr(self.model, "module") and hasattr(self.model.module, "classifier")):
            classifier = self.model.module.classifier if hasattr(self.model, "module") else self.model.classifier
            for p in classifier.parameters():
                p.requires_grad = True

        batch_size = self.unfrozen_batch_size if not freeze else self.batch_size
        # when distributed, batch_size is per-process; ensure user knows to set appropriate batch_size for each GPU
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

            # save checkpoint (latest) only on rank 0
            if save_every_epoch and self.rank == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
                # if DDP, save module.state_dict
                model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            # save best model by val acc
            if use_validation and val_acc > best_val_acc and self.rank == 0:
                best_val_acc = val_acc
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save(model_state, best_path)
                print(f"New best model saved to {best_path} (val_acc={val_acc*100:.2f}%)")

            if hasattr(self, "scheduler") and self.rank == 0:
                self.scheduler.step()
                print(f"[Scheduler] lr now = {self.scheduler.get_last_lr()[0]:.6f}")
            elif hasattr(self, "scheduler"):
                # still step scheduler in all ranks to keep optimizer state consistent if needed
                self.scheduler.step()

            elapsed = time.time() - start_ts
            if self.rank == 0:
                print(f"Epoch {epoch} finished in {elapsed:.1f}s")

    def evaluate(self, split='test'):
        avg_loss, avg_acc = self._run_epoch(split)
        if self.rank == 0:
            print(f"{split.capitalize()} - loss: {avg_loss:.4f}, acc: {avg_acc*100:.2f}%")

        # collect predictions and labels for full report
        dataloader = self.dataloaders[split]
        # We will collect local preds/labels, then gather to rank 0 for full report
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            pbar = tqdm(dataloader, file=sys.stdout, dynamic_ncols=True) if self.rank == 0 else dataloader
            for batch in pbar:
                video_data, labels = batch['vision'], batch['label']
                pixel_values = self.video_preprocessor.preprocess_batch(video_data).to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(pixel_values=pixel_values, interpolate_pos_encoding=True)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # gather results to rank 0
        if self.distributed:
            # serialize arrays as tensors for all_gather
            local_preds = torch.tensor(all_preds, dtype=torch.int64, device=self.device)
            local_labels = torch.tensor(all_labels, dtype=torch.int64, device=self.device)
            # lengths
            local_len = torch.tensor([local_preds.numel()], dtype=torch.int64, device=self.device)
            lengths = [torch.zeros_like(local_len) for _ in range(self.world_size)]
            dist.all_gather(lengths, local_len)
            max_len = int(max([int(x.item()) for x in lengths]))

            # pad to max_len
            def pad_tensor(t, max_len):
                if t.numel() < max_len:
                    pad = torch.zeros(max_len - t.numel(), dtype=t.dtype, device=t.device)
                    return torch.cat([t, pad], dim=0)
                return t

            padded_preds = pad_tensor(local_preds, max_len)
            padded_labels = pad_tensor(local_labels, max_len)

            gathered_preds = [torch.zeros_like(padded_preds) for _ in range(self.world_size)]
            gathered_labels = [torch.zeros_like(padded_labels) for _ in range(self.world_size)]
            dist.all_gather(gathered_preds, padded_preds)
            dist.all_gather(gathered_labels, padded_labels)

            if self.rank == 0:
                final_preds = []
                final_labels = []
                for i in range(self.world_size):
                    ln = int(lengths[i].item())
                    if ln > 0:
                        final_preds.extend(gathered_preds[i].cpu().numpy()[:ln].tolist())
                        final_labels.extend(gathered_labels[i].cpu().numpy()[:ln].tolist())
                all_preds = np.array(final_preds)
                all_labels = np.array(final_labels)
            else:
                all_preds = None
                all_labels = None
        else:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

        if self.rank == 0:
            f1 = f1_score(all_labels, all_preds, average='weighted')
            print("Classification Report:")
            print(classification_report(all_labels, all_preds))
            print("Confusion Matrix:")
            print(confusion_matrix(all_labels, all_preds))
            return avg_acc, f1
        else:
            return None, None


def main():
    # setup distributed if launched with torchrun/launch
    distributed = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()

    if rank == 0:
        print("\n" + "=" * 80)
        print("Cross-Subject ViViT Training (DDP-enabled)")
        print("=" * 80)

    # paths & config (keep same as your script)
    data_root_dict = {
        'audio': '/data2/mingzhi/BCI/dataset/EAV_old/Audio',
        'eeg': '/data2/mingzhi/BCI/dataset/EAV_old/EEG',
        'vision': '/data2/mingzhi/BCI/dataset/EAV_old/Vision'
    }
    model_path = '/data2/mingzhi/BCI/WMZ_BCI/pretrained_model/ViVit'
    csv_path = "/data2/mingzhi/BCI/dataset/EAV_old/subjects.csv"

    if rank == 0 and not os.path.exists(model_path):
        print(f"✗ Model path not found: {model_path}")
        # allow other ranks to exit gracefully
        if distributed:
            dist.barrier()
        return

    trainer = None
    try:
        trainer = CrossSubjectViViTTrainer(
            data_root_dict=data_root_dict,
            model_path=model_path,
            num_labels=5,
            batch_size=32,              # per-process batch size when not distributed; when distributed this is per-process
            unfrozen_batch_size=4,
            num_workers=32,
            random_seed=2024,
            subject_csv_path=csv_path if os.path.exists(csv_path) else None,
            lr=1e-3,
            checkpoint_dir="./checkpoints",
            distributed=distributed
        )

        # Phase 1: frozen training
        if rank == 0:
            print("\nPHASE 1: Frozen Feature Training")
        trainer.train(epochs=30, freeze=True, use_validation=True, save_every_epoch=True)

        # Clear GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Phase 2: unfrozen fine-tune
        if rank == 0:
            print("\nPHASE 2: Unfrozen Fine-tuning")
        trainer.train(epochs=0, freeze=False, use_validation=True, save_every_epoch=True)

        if rank == 0:
            print("\nFINAL EVALUATION on test set")
        acc_f1 = trainer.evaluate(split='test')
        if rank == 0:
            acc, f1 = acc_f1
            if acc is not None:
                print(f"Final Test Acc: {acc*100:.2f}%, F1: {f1:.4f}")

                # save summary (only rank 0)
                results = {'test_acc': acc, 'test_f1': f1,
                           'train_subjects': getattr(trainer.dataset_loader, 'train_subjects', None),
                           'val_subjects': getattr(trainer.dataset_loader, 'val_subjects', None),
                           'test_subjects': getattr(trainer.dataset_loader, 'test_subjects', None)}
                with open('cross_subject_vivit_results.pkl', 'wb') as f:
                    pickle.dump(results, f)
                print("Saved results to cross_subject_vivit_results.pkl")

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        if trainer is not None:
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if distributed and is_dist_available_and_initialized():
            # ensure all processes exit together
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
