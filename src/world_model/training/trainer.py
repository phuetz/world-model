"""Trainer — boucle d'entraînement du WorldModel."""
from __future__ import annotations
import os
import time
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ..config.config import WorldModelConfig
from ..models.world_model import WorldModel


class Trainer:
    """Entraîne le WorldModel sur des transitions (obs_t, a_t, obs_tp1, r_t, done_t)."""

    def __init__(
        self,
        model: WorldModel,
        cfg: WorldModelConfig,
        log_dir: str = "runs/world_model",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)

        n_gpus = torch.cuda.device_count() if self.device.type == "cuda" else 0
        if n_gpus > 1:
            self.model = nn.DataParallel(model)
            print(f"DataParallel actif sur {n_gpus} GPUs : {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
        else:
            self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        self.writer = SummaryWriter(log_dir)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        print(f"Device : {self.device}")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paramètres entraînables : {n_params:,}")

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals: Dict[str, float] = {}
        steps = 0

        for obs_t, action_t, obs_tp1, reward_t, done_t in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            obs_t    = obs_t.to(self.device)
            action_t = action_t.to(self.device)
            obs_tp1  = obs_tp1.to(self.device)
            reward_t = reward_t.to(self.device)
            done_t   = done_t.to(self.device)

            self.optimizer.zero_grad()
            losses = self.model(obs_t, action_t, obs_tp1, reward_t, done_t)
            # DataParallel gather concat sur dim 0 → mean pour réduire à un scalaire
            losses = {k: v.mean() for k, v in losses.items()}
            losses["loss_total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            steps += 1

        return {k: v / steps for k, v in totals.items()}

    def fit(self, loader: DataLoader) -> None:
        global_step = 0
        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            metrics = self.train_epoch(loader, epoch)
            elapsed = time.time() - t0

            # TensorBoard
            for k, v in metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

            print(
                f"Epoch {epoch:3d}/{self.cfg.max_epochs} | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                + f" | {elapsed:.1f}s"
            )

            # Checkpoint toutes les 10 epochs (déballe DataParallel pour state_dict propre)
            if epoch % 10 == 0:
                path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
                core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                torch.save({
                    "epoch": epoch,
                    "model_state": core_model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "metrics": metrics,
                }, path)
                print(f"  >> Checkpoint saved: {path}")

        self.writer.close()
        print("Entraînement terminé.")
