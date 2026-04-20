"""LatentDynamicsModel — prédit z_{t+1} à partir de z_t et a_t."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config.config import WorldModelConfig


class LatentDynamicsModel(nn.Module):
    """Modèle de transition latent : f(z_t, a_enc_t) → z_{t+1}_pred."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        # Têtes auxiliaires optionnelles
        self.reward_head = nn.Linear(cfg.latent_dim, 1) if cfg.use_reward_head else None
        self.done_head = nn.Linear(cfg.latent_dim, 1) if cfg.use_done_head else None

    def forward(self, z: torch.Tensor, a_enc: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim), a_enc: (B, latent_dim) → z_next: (B, latent_dim)"""
        return self.net(torch.cat([z, a_enc], dim=-1))

    def predict_reward(self, z_next: torch.Tensor) -> torch.Tensor | None:
        """Prédit la récompense depuis l'état latent suivant."""
        return self.reward_head(z_next).squeeze(-1) if self.reward_head else None

    def predict_done(self, z_next: torch.Tensor) -> torch.Tensor | None:
        """Prédit la fin d'épisode depuis l'état latent suivant."""
        return self.done_head(z_next).squeeze(-1) if self.done_head else None
