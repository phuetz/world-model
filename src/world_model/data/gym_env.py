"""Collecte de transitions depuis un environnement Gymnasium."""
from __future__ import annotations
from typing import Callable
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from ..config.config import WorldModelConfig


def _heuristic_carracing(step: int, env, rng: np.random.Generator) -> np.ndarray:
    """Politique heuristique pour CarRacing : accélère, oscille en steering.
    action = (steering ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]).
    Donne des trajectoires plus structurées qu'une politique random."""
    steering = 0.6 * math.sin(step * 0.05) + 0.2 * rng.standard_normal()
    steering = float(np.clip(steering, -1.0, 1.0))
    gas = 0.4 + 0.2 * rng.random()
    brake = 0.0
    return np.array([steering, gas, brake], dtype=np.float32)


def _mixed_carracing(step: int, env, rng: np.random.Generator) -> np.ndarray:
    """Mélange 50/50 par step : capture haute fréquence (random) + structure (heuristique)."""
    if rng.random() < 0.5:
        return env.action_space.sample()
    return _heuristic_carracing(step, env, rng)


POLICIES: dict[str, Callable] = {
    "heuristic": _heuristic_carracing,
    "mixed": _mixed_carracing,
}


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
    """Collecte n_samples transitions depuis un env Gymnasium.

    policy='random' : env.action_space.sample()
    policy='heuristic' : politique scriptée (cf. POLICIES)
    """

    def __init__(self, cfg: WorldModelConfig, env_id: str, n_samples: int,
                 seed: int = 0, policy: str = "random") -> None:
        import gymnasium as gym

        self.cfg = cfg
        C, H, W = cfg.obs_shape

        env = gym.make(env_id)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        rng = np.random.default_rng(seed)
        policy_fn = POLICIES.get(policy)

        obs_t_buf   = torch.empty(n_samples, C, H, W)
        obs_tp1_buf = torch.empty(n_samples, C, H, W)
        action_buf  = torch.empty(n_samples, cfg.action_dim)
        reward_buf  = torch.empty(n_samples)
        done_buf    = torch.empty(n_samples)

        obs, _ = env.reset(seed=seed)
        episode_step = 0
        i = 0
        with tqdm(total=n_samples, desc=f"Collect {env_id} ({policy})") as pbar:
            while i < n_samples:
                if policy_fn is not None:
                    action = policy_fn(episode_step, env, rng)
                else:
                    action = env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                obs_t_buf[i]   = _preprocess_obs(np.asarray(obs), H, W)
                obs_tp1_buf[i] = _preprocess_obs(np.asarray(next_obs), H, W)
                action_buf[i]  = _pad_action(action, cfg.action_dim)
                reward_buf[i]  = float(reward)
                done_buf[i]    = float(done)

                i += 1
                episode_step += 1
                pbar.update(1)

                if done:
                    obs, _ = env.reset()
                    episode_step = 0
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
    cfg: WorldModelConfig, env_id: str, n_samples: int, seed: int = 0,
    policy: str = "random",
) -> DataLoader:
    dataset = GymTransitionDataset(cfg, env_id, n_samples, seed=seed, policy=policy)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
