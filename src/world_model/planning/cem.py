"""CEM planner — Cross-Entropy Method en latent.

Optimise une séquence d'actions (H, action_dim) en imaginant des rollouts
dans le latent du WorldModel. Score = somme des rewards prédits par la
reward_head, optionnellement pondéré par (1 - done_pred).

Usage typique (MPC) : appeler plan(z_init) à chaque step, prendre la première
action, observer, ré-encoder, replanner.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from ..models.world_model import WorldModel


@dataclass
class CEMConfig:
    horizon: int = 12          # H steps imaginés
    n_samples: int = 512       # candidats par itération
    n_elites: int = 64         # top-K conservés
    n_iterations: int = 4      # itérations de raffinement
    init_std: float = 0.5      # écart-type initial autour de la moyenne
    min_std: float = 0.05      # plancher d'écart-type (évite collapse Gauss)
    momentum: float = 0.1      # mélange ancien/nouveau mean (0 = pas de momentum)
    action_low: float = -1.0
    action_high: float = 1.0
    done_penalty: float = 5.0  # pénalité ajoutée sur done prédit


class CEMPlanner:
    """CEM au-dessus du WorldModel. Tout vectorisé sur GPU."""

    def __init__(self, model: WorldModel, action_dim: int,
                 cfg: Optional[CEMConfig] = None,
                 device: Optional[torch.device] = None) -> None:
        self.model = model.eval()
        self.action_dim = action_dim
        self.cfg = cfg or CEMConfig()
        self.device = device or next(model.parameters()).device
        # Mean conservée d'un appel sur l'autre pour MPC (warm start)
        self.last_mean: Optional[torch.Tensor] = None

    @torch.no_grad()
    def plan(self, z_init: torch.Tensor) -> torch.Tensor:
        """z_init: (latent_dim,) ou (1, latent_dim). Retourne actions (H, action_dim)."""
        cfg = self.cfg
        if z_init.ndim == 1:
            z_init = z_init.unsqueeze(0)
        H, N, K = cfg.horizon, cfg.n_samples, cfg.n_elites

        # Warm start : décale la séquence précédente d'un step (MPC)
        if self.last_mean is not None and self.last_mean.shape[0] == H:
            mean = torch.cat([self.last_mean[1:], self.last_mean[-1:].clone()], dim=0)
        else:
            mean = torch.zeros(H, self.action_dim, device=self.device)
        std = torch.full((H, self.action_dim), cfg.init_std, device=self.device)

        for _ in range(cfg.n_iterations):
            noise = torch.randn(N, H, self.action_dim, device=self.device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(
                cfg.action_low, cfg.action_high
            )
            scores = self._score(z_init, actions)  # (N,)

            elites_idx = scores.topk(K).indices
            elites = actions[elites_idx]            # (K, H, A)
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0).clamp_min(cfg.min_std)

            mean = (1 - cfg.momentum) * new_mean + cfg.momentum * mean
            std = new_std

        self.last_mean = mean
        return mean

    @torch.no_grad()
    def _score(self, z_init: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z_init: (1, D). actions: (N, H, A). Retourne (N,) = somme rewards prédits."""
        N, H, _ = actions.shape
        z = z_init.expand(N, -1).contiguous()  # (N, D)
        total_reward = torch.zeros(N, device=self.device)
        alive = torch.ones(N, device=self.device)
        for h in range(H):
            a = actions[:, h]
            z = self.model.predict_next(z, a)
            r = self.model.dynamics.predict_reward(z) if self.model.dynamics.reward_head else None
            d = self.model.dynamics.predict_done(z) if self.model.dynamics.done_head else None
            if r is not None:
                total_reward = total_reward + alive * r
            if d is not None:
                done_p = torch.sigmoid(d)
                total_reward = total_reward - self.cfg.done_penalty * alive * done_p
                alive = alive * (1.0 - done_p)
        return total_reward
