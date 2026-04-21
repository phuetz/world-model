"""Collecte de transitions depuis un environnement Gymnasium (politique aléatoire)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ..config.config import WorldModelConfig


def _preprocess_obs(obs: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    """(H, W, C) uint8 → (C, target_h, target_w) float32 dans [0, 1]."""
    t = torch.from_numpy(obs).float() / 255.0
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0)


def _pad_action(action: np.ndarray | int | float, target_dim: int) -> torch.Tensor:
    """Action env → vecteur de dim target_dim (pad zéros si plus petite)."""
    a = np.atleast_1d(np.asarray(action, dtype=np.float32))
    if a.shape[0] < target_dim:
        a = np.concatenate([a, np.zeros(target_dim - a.shape[0], dtype=np.float32)])
    elif a.shape[0] > target_dim:
        a = a[:target_dim]
    return torch.from_numpy(a)


class GymTransitionDataset(Dataset):
    """Collecte n_samples transitions depuis un env Gymnasium avec une politique aléatoire."""

    def __init__(self, cfg: WorldModelConfig, env_id: str, n_samples: int, seed: int = 0) -> None:
        import gymnasium as gym

        self.cfg = cfg
        C, H, W = cfg.obs_shape

        env = gym.make(env_id)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        obs_t_buf   = torch.empty(n_samples, C, H, W)
        obs_tp1_buf = torch.empty(n_samples, C, H, W)
        action_buf  = torch.empty(n_samples, cfg.action_dim)
        reward_buf  = torch.empty(n_samples)
        done_buf    = torch.empty(n_samples)

        obs, _ = env.reset(seed=seed)
        i = 0
        with tqdm(total=n_samples, desc=f"Collect {env_id}") as pbar:
            while i < n_samples:
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                obs_t_buf[i]   = _preprocess_obs(np.asarray(obs), H, W)
                obs_tp1_buf[i] = _preprocess_obs(np.asarray(next_obs), H, W)
                action_buf[i]  = _pad_action(action, cfg.action_dim)
                reward_buf[i]  = float(reward)
                done_buf[i]    = float(done)

                i += 1
                pbar.update(1)

                if done:
                    obs, _ = env.reset()
                else:
                    obs = next_obs

        env.close()

        self.obs_t   = obs_t_buf
        self.obs_tp1 = obs_tp1_buf
        self.actions = action_buf
        self.rewards = reward_buf
        self.dones   = done_buf

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


def make_gym_dataloader(
    cfg: WorldModelConfig, env_id: str, n_samples: int, seed: int = 0
) -> DataLoader:
    dataset = GymTransitionDataset(cfg, env_id, n_samples, seed=seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
