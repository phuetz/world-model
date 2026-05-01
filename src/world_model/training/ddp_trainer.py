"""DDP Trainer V3 — distributed data parallel sur Windows (backend gloo) + bf16.

Pattern :
- init_process_group("gloo") — NCCL absent sur Win11
- DistributedSampler avec set_epoch
- bf16 autocast (pas de GradScaler nécessaire en bf16)
- AdamW + cosine schedule + warmup linéaire 1000 steps
- Sauvegarde rank=0 only, garde N derniers checkpoints
- Warmup d'epochs : N epochs en forward_step (1-step) avant rollout T-step

Compatible avec un Dataset 4-tuple (obs_seq, action_seq, reward_seq, done_seq) en mode rollout,
ou 5-tuple (obs_t, action_t, obs_tp1, reward_t, done_t) en mode 1-step (warmup).
"""
from __future__ import annotations
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config.config import WorldModelConfig
from ..models.world_model import WorldModel


def setup_ddp(backend: str = "gloo") -> int:
    """Init process group depuis env vars torchrun. Retourne local_rank."""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class DDPTrainer:
    def __init__(
        self,
        model: WorldModel,
        cfg: WorldModelConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        log_dir: str = "runs/v3_video",
        checkpoint_dir: str = "checkpoints_v3_video",
        warmup_steps: int = 1000,
        keep_last_n: int = 3,
        save_every_epochs: int = 5,
        backend: str = "gloo",
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.cfg = cfg
        self.local_rank = setup_ddp(backend)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = (dist.get_rank() == 0) if dist.is_initialized() else True
        self.device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")

        model.to(self.device)
        # find_unused_parameters=True : nécessaire car le warmup 1-step n'active pas
        # certaines positions du transformer dynamique (learned positional embed sur T-1
        # positions inutilisées au step k=1).
        if self.world_size > 1 and torch.cuda.is_available():
            self.model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], find_unused_parameters=True,
            )
        elif self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        steps_per_epoch = max(1, len(train_loader))
        self.total_steps = steps_per_epoch * cfg.max_epochs
        self.warmup_steps = warmup_steps

        self.checkpoint_dir = Path(checkpoint_dir)
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        self.keep_last_n = keep_last_n
        self.save_every_epochs = save_every_epochs
        self.amp_dtype = amp_dtype if cfg.use_amp else None

        if self.is_main:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[DDP rank0] device={self.device} world_size={self.world_size} params={n_params:,} "
                  f"warmup_epochs={cfg.rollout_warmup_epochs} amp={'bf16' if self.amp_dtype else 'fp32'}")

    # ---------------------------------------------------------------- training

    def _step(self, batch, use_rollout: bool) -> Dict[str, torch.Tensor]:
        batch = [t.to(self.device, non_blocking=True) for t in batch]
        if self.amp_dtype is not None:
            ctx = torch.autocast(device_type="cuda", dtype=self.amp_dtype)
        else:
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, *a): return None
            ctx = _NullCtx()

        with ctx:
            if use_rollout and len(batch) == 4:
                obs_seq, action_seq, reward_seq, done_seq = batch
                losses = self.model(obs_seq, action_seq, reward_seq, done_seq)
            elif len(batch) == 4:
                # Warmup : tronque la séquence à K=1 et passe via le wrapper DDP
                # (forward_rollout avec K=1 == forward_step en termes de loss, mais
                # en restant dans le chemin DDP-tracé pour synchroniser les grads).
                obs_seq, action_seq, reward_seq, done_seq = batch
                obs_seq = obs_seq[:, :2]            # (B, 2, C, H, W)
                action_seq = action_seq[:, :1]      # (B, 1, action_dim)
                reward_seq = reward_seq[:, :1]
                done_seq = done_seq[:, :1]
                losses = self.model(obs_seq, action_seq, reward_seq, done_seq)
            else:
                # Dataset 5-tuple
                losses = self.model(*batch)
        return losses

    def train_epoch(self, epoch: int, global_step: int) -> tuple[Dict[str, float], int]:
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        use_rollout = epoch > self.cfg.rollout_warmup_epochs
        totals: Dict[str, float] = {}
        steps = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            losses = self._step(batch, use_rollout=use_rollout)
            losses_mean = {k: v.mean() for k, v in losses.items()}
            loss_total = losses_mean["loss_total"]
            loss_total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # LR schedule
            lr = cosine_with_warmup(global_step, self.warmup_steps, self.total_steps, self.cfg.learning_rate)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.step()

            for k, v in losses_mean.items():
                totals[k] = totals.get(k, 0.0) + float(v.item())
            steps += 1
            global_step += 1

            if self.is_main and global_step % 50 == 0 and self.writer is not None:
                for k, v in losses_mean.items():
                    self.writer.add_scalar(f"train/{k}", float(v.item()), global_step)
                self.writer.add_scalar("train/lr", lr, global_step)

        avg = {k: v / max(1, steps) for k, v in totals.items()}
        return avg, global_step

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals: Dict[str, float] = {}
        steps = 0
        for batch in self.val_loader:
            losses = self._step(batch, use_rollout=True)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.mean().item())
            steps += 1
        return {k: v / max(1, steps) for k, v in totals.items()}

    # ---------------------------------------------------------------- checkpoint

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        if not self.is_main:
            return
        path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"
        core_model = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        torch.save(
            {
                "epoch": epoch,
                "model_state": core_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg.__dict__,
            },
            path,
        )
        # Rotation
        ckpts = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        for old in ckpts[:-self.keep_last_n] if len(ckpts) > self.keep_last_n else []:
            try:
                old.unlink()
            except OSError:
                pass
        print(f"  >> checkpoint saved: {path}")

    # ---------------------------------------------------------------- fit

    def fit(self, max_steps: Optional[int] = None) -> None:
        global_step = 0
        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            metrics, global_step = self.train_epoch(epoch, global_step)
            elapsed = time.time() - t0

            if self.is_main:
                line = (
                    f"Epoch {epoch:3d}/{self.cfg.max_epochs} | "
                    + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                    + f" | {elapsed:.1f}s"
                )
                if self.cfg.rollout_warmup_epochs and epoch <= self.cfg.rollout_warmup_epochs:
                    line += " [warmup 1-step]"
                print(line, flush=True)

            if epoch % self.save_every_epochs == 0:
                val_metrics = self.validate() if self.val_loader is not None else {}
                if self.is_main and self.writer is not None:
                    for k, v in val_metrics.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                self._save_checkpoint(epoch, {**metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

            if max_steps is not None and global_step >= max_steps:
                if self.is_main:
                    print(f"[fit] max_steps {max_steps} reached, stopping early")
                break

        if self.is_main and self.writer is not None:
            self.writer.close()

    def close(self) -> None:
        cleanup_ddp()
