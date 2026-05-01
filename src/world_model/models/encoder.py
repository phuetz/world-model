"""ObservationEncoder et ActionEncoder — encode observations et actions dans l'espace latent."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config.config import WorldModelConfig


class ObservationEncoder(nn.Module):
    """Encode une image (C, H, W) en vecteur latent z de dimension latent_dim."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        C, H, W = cfg.obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),   # H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # H/8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# H/16
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calcule la taille de sortie conv
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy).shape[1]

        self.proj = nn.Linear(conv_out, cfg.latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) → z: (B, latent_dim)"""
        return self.proj(self.conv(obs))


class ObservationEncoderConv5(nn.Module):
    """Encoder Conv5 pour V3 (256×256). 5 convs stride-2 → 8×8 feature map → Linear → latent_dim.

    Différence avec ObservationEncoder (Conv4) : un étage de stride-2 de plus,
    nécessaire pour ne pas exploser la projection finale en 256×256.
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        C, H, W = cfg.obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1),    # H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # H/8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # H/16
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), # H/32
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy).shape[1]
        self.proj = nn.Linear(conv_out, cfg.latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(obs))


def build_observation_encoder(cfg: WorldModelConfig) -> nn.Module:
    """Factory selon cfg.encoder_type."""
    if cfg.encoder_type == "conv5":
        return ObservationEncoderConv5(cfg)
    # default conv (V1/V2)
    return ObservationEncoder(cfg)


class ActionEncoder(nn.Module):
    """Projette un vecteur action dans l'espace utilisé par le modèle de dynamique."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """action: (B, action_dim) → a_enc: (B, latent_dim)"""
        return self.net(action)
