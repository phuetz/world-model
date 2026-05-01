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


class LatentDynamicsTransformer(nn.Module):
    """V3 dynamics : transformer encoder causal pre-norm.

    Input séquence de paires (z_t, a_enc_t) concat → token d_model=2*latent_dim
    projeté à latent_dim. Output une séquence de prédictions z_{t+1} pour chaque t.
    Compat 1-step via une séquence de longueur 1 (pour CEM / warmup).
    """

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.latent_dim = cfg.latent_dim
        self.seq_len = cfg.seq_len

        d_model = cfg.latent_dim
        n_heads = 8
        n_layers = 4
        d_ff = d_model * 4
        max_len = max(cfg.seq_len, 32)  # marge pour rollout plus long en eval

        self.input_proj = nn.Linear(cfg.latent_dim * 2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, cfg.latent_dim)

        # Causal mask (re-utilisé au forward, agrandi à la demande si T plus grand)
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1),
            persistent=False,
        )

        # Têtes auxiliaires (mêmes signatures que LatentDynamicsModel pour compat)
        self.reward_head = nn.Linear(cfg.latent_dim, 1) if cfg.use_reward_head else None
        self.done_head = nn.Linear(cfg.latent_dim, 1) if cfg.use_done_head else None

    def _step(self, z_seq: torch.Tensor, a_seq: torch.Tensor) -> torch.Tensor:
        """z_seq, a_seq : (B, T, latent_dim) → z_pred : (B, T, latent_dim)."""
        B, T, _ = z_seq.shape
        x = torch.cat([z_seq, a_seq], dim=-1)         # (B, T, 2*latent)
        x = self.input_proj(x)                         # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]
        if T > self._causal_mask.shape[0]:
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
        else:
            mask = self._causal_mask[:T, :T]
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.output_proj(x)

    def forward(self, z: torch.Tensor, a_enc: torch.Tensor) -> torch.Tensor:
        """Compat 1-step (signature LatentDynamicsModel).

        z, a_enc : (B, latent_dim) → z_next : (B, latent_dim).
        """
        z_seq = z.unsqueeze(1)
        a_seq = a_enc.unsqueeze(1)
        out = self._step(z_seq, a_seq)
        return out.squeeze(1)

    def forward_sequence(
        self, z_seq: torch.Tensor, a_enc_seq: torch.Tensor
    ) -> torch.Tensor:
        """Rollout batch parallèle.

        z_seq    : (B, T, latent_dim) — états de contexte z_0..z_{T-1}
        a_enc_seq: (B, T, latent_dim) — actions encodées
        Retourne : (B, T, latent_dim) — prédictions z_1..z_T
        """
        return self._step(z_seq, a_enc_seq)

    def predict_reward(self, z_next: torch.Tensor) -> torch.Tensor | None:
        return self.reward_head(z_next).squeeze(-1) if self.reward_head else None

    def predict_done(self, z_next: torch.Tensor) -> torch.Tensor | None:
        return self.done_head(z_next).squeeze(-1) if self.done_head else None


def build_dynamics(cfg: WorldModelConfig) -> nn.Module:
    """Factory selon cfg.dynamics_type."""
    if cfg.dynamics_type == "transformer":
        return LatentDynamicsTransformer(cfg)
    return LatentDynamicsModel(cfg)
