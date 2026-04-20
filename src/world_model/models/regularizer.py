"""IsotropicLatentRegularizer — stabilise l'espace latent, évite le collapse."""
from __future__ import annotations
import torch
import torch.nn as nn
from ..config.config import WorldModelConfig


class IsotropicLatentRegularizer(nn.Module):
    """
    Régularisation inspirée de VICReg :
    - Pénalité variance : force chaque dimension à rester informative
    - Pénalité covariance : découple les dimensions (hors-diagonale → 0)
    - Pénalité moyenne : centre les représentations autour de 0
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.lambda_var = cfg.lambda_var
        self.lambda_cov = cfg.lambda_cov
        self.lambda_mean = cfg.lambda_mean

    def compute(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: (B, latent_dim)
        Retourne le scalaire de régularisation.
        """
        B, D = latents.shape
        z = latents - latents.mean(dim=0)  # centre

        # Variance : chaque dim doit avoir std ≥ 1
        std = z.std(dim=0)
        loss_var = torch.mean(torch.relu(1.0 - std))

        # Covariance : hors-diagonale → 0
        cov = (z.T @ z) / (B - 1)
        off_diag = cov - torch.diag(cov.diag())
        loss_cov = (off_diag ** 2).sum() / D

        # Moyenne : centrage autour de 0
        loss_mean = (latents.mean(dim=0) ** 2).mean()

        return (
            self.lambda_var  * loss_var
            + self.lambda_cov  * loss_cov
            + self.lambda_mean * loss_mean
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.compute(latents)
