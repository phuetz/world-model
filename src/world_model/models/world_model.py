"""WorldModel — orchestrateur principal JEPA."""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from .encoder import ObservationEncoder, ActionEncoder
from .dynamics import LatentDynamicsModel
from .regularizer import IsotropicLatentRegularizer
from ..config.config import WorldModelConfig


class WorldModel(nn.Module):
    """
    World Model inspiré de JEPA / LeWorldModel.

    Prédiction dans l'espace latent (pas reconstruction pixel) :
      z_t     = ObservationEncoder(obs_t)
      a_enc_t = ActionEncoder(action_t)
      z_pred  = LatentDynamicsModel(z_t, a_enc_t)
      z_target= ObservationEncoder(obs_{t+1})  [stop-gradient]
      loss    = MSE(z_pred, z_target) + regularisation isotrope
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_encoder   = ObservationEncoder(cfg)
        self.action_encoder = ActionEncoder(cfg)
        self.dynamics      = LatentDynamicsModel(cfg)
        self.regularizer   = IsotropicLatentRegularizer(cfg)

    def forward_step(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_tp1: torch.Tensor,
        reward_t: torch.Tensor | None = None,
        done_t: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Un pas de prédiction latente.
        obs_t, obs_tp1 : (B, C, H, W)
        action_t       : (B, action_dim)
        Retourne un dict de losses.
        """
        # Encode contexte et cible
        z_t       = self.obs_encoder(obs_t)
        a_enc_t   = self.action_encoder(action_t)

        with torch.no_grad():
            z_target = self.obs_encoder(obs_tp1)   # stop-gradient sur la cible

        # Prédiction
        z_pred = self.dynamics(z_t, a_enc_t)

        # Loss principale : MSE dans l'espace latent
        loss_pred = nn.functional.mse_loss(z_pred, z_target)

        # Régularisation
        loss_reg = self.regularizer(z_pred)

        losses: Dict[str, torch.Tensor] = {
            "loss_pred": loss_pred,
            "loss_reg":  loss_reg,
            "loss_total": loss_pred + loss_reg,
        }

        # Têtes auxiliaires
        if self.cfg.use_reward_head and reward_t is not None:
            r_pred = self.dynamics.predict_reward(z_pred)
            losses["loss_reward"] = nn.functional.mse_loss(r_pred, reward_t)
            losses["loss_total"] = losses["loss_total"] + losses["loss_reward"]

        if self.cfg.use_done_head and done_t is not None:
            d_pred = self.dynamics.predict_done(z_pred)
            losses["loss_done"] = nn.functional.binary_cross_entropy_with_logits(d_pred, done_t.float())
            losses["loss_total"] = losses["loss_total"] + losses["loss_done"]

        # unsqueeze pour permettre à DataParallel de gather les scalaires (concat dim 0)
        return {k: v.unsqueeze(0) for k, v in losses.items()}

    def forward(self, *args, **kwargs):
        return self.forward_step(*args, **kwargs)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode une observation en vecteur latent (inference)."""
        return self.obs_encoder(obs)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Prédit le prochain état latent (planification)."""
        a_enc = self.action_encoder(action)
        return self.dynamics(z, a_enc)
