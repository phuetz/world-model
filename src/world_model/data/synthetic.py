"""Générateur de données synthétiques pour valider l'architecture V1."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from ..config.config import WorldModelConfig


class SyntheticTransitionDataset(Dataset):
    """
    Dataset de transitions (obs_t, action_t, obs_tp1, reward, done) synthétiques.
    Permet de valider l'architecture sans environnement réel.
    """

    def __init__(self, cfg: WorldModelConfig, n_samples: int = 10000) -> None:
        self.cfg = cfg
        C, H, W = cfg.obs_shape
        self.obs_t   = torch.rand(n_samples, C, H, W)
        self.obs_tp1 = torch.rand(n_samples, C, H, W)
        self.actions = torch.rand(n_samples, cfg.action_dim)
        self.rewards = torch.randn(n_samples)
        self.dones   = torch.bernoulli(torch.full((n_samples,), 0.05))

    def __len__(self) -> int:
        return len(self.obs_t)

    def __getitem__(self, idx: int):
        return (
            self.obs_t[idx],
            self.actions[idx],
            self.obs_tp1[idx],
            self.rewards[idx],
            self.dones[idx],
        )


def make_dataloader(cfg: WorldModelConfig, n_samples: int = 10000) -> DataLoader:
    dataset = SyntheticTransitionDataset(cfg, n_samples)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
