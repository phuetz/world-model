"""V4 — dataset Gymnasium env réel pour world-model JEPA.

Mime l'API VideoClipDataset (V3 video) mais collecte des trajectoires
depuis un env Gymnasium avec des actions vraies (continuous ou discrete),
au lieu de clips vidéo passifs avec optical flow comme proxy.

Motivation : sur dataset video passif (V3 SVD-XT, V3.1 Wan), l'effective
rank du latent stagne ou s'effondre (2.9% V3, 0.28% V3.1) car l'action
proxy 4D (Farneback) est dégénérée sur scènes peu mobiles. V1.8 sur
CarRacing (actions vraies) atteignait 8% rank "for free". V4 reprend
l'archi V3 (Conv5/Conv4 + Transformer dynamique) mais avec env Gymnasium.

Layout : tout en RAM (pas de cache disque). Pour 10k transitions à
64×64×3 uint8 ≈ 120 MB — gérable. Si scaling au-delà, cacher sur disque
au format JPEG comme V3.

Item __getitem__ aligné sur VideoClipDataset :
    obs_seq    : (T+1, C, H, W) float32 ∈ [-1, 1]
    action_seq : (T, action_dim) float32
    reward_seq : (T,) float32
    done_seq   : (T,) float32
"""
from __future__ import annotations
from typing import Callable, List, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config.config import WorldModelConfig


def _heuristic_lunarlander(step: int, env, rng: np.random.Generator) -> np.ndarray:
    """Politique heuristique LunarLanderContinuous : main engine + lateral oscillation.
    action = (main ∈ [-1, 1] (>0 fires main), lateral ∈ [-1, 1])."""
    main = 0.4 + 0.3 * math.sin(step * 0.1) + 0.2 * rng.standard_normal()
    main = float(np.clip(main, -1.0, 1.0))
    lateral = 0.3 * math.sin(step * 0.07) + 0.1 * rng.standard_normal()
    lateral = float(np.clip(lateral, -1.0, 1.0))
    return np.array([main, lateral], dtype=np.float32)


def _heuristic_carracing(step: int, env, rng: np.random.Generator) -> np.ndarray:
    """Politique heuristique CarRacing (port depuis gym_env.py)."""
    steering = 0.6 * math.sin(step * 0.05) + 0.2 * rng.standard_normal()
    steering = float(np.clip(steering, -1.0, 1.0))
    gas = 0.4 + 0.2 * rng.random()
    brake = 0.0
    return np.array([steering, gas, brake], dtype=np.float32)


HEURISTICS: dict[str, Callable] = {
    "LunarLanderContinuous-v3": _heuristic_lunarlander,
    "CarRacing-v3": _heuristic_carracing,
}


def _frame_to_tensor(frame: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    """(H, W, 3) uint8 → (3, target_h, target_w) float32 ∈ [-1, 1].

    Aligné sur video_dataset._load_frame : / 127.5 - 1.0.
    """
    t = torch.from_numpy(frame.copy()).float() / 127.5 - 1.0  # [-1, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)
    if t.shape[-2] != target_h or t.shape[-1] != target_w:
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0)


def _pad_action(action: np.ndarray | int | float, target_dim: int) -> np.ndarray:
    a = np.atleast_1d(np.asarray(action, dtype=np.float32))
    if a.shape[0] < target_dim:
        a = np.concatenate([a, np.zeros(target_dim - a.shape[0], dtype=np.float32)])
    elif a.shape[0] > target_dim:
        a = a[:target_dim]
    return a


class GymVideoDataset(Dataset):
    """Collecte n_episodes × max_episode_len transitions depuis un env Gymnasium
    en gardant les rendus RGB, et expose des fenêtres T+1 frames pour le trainer V3/V4.

    Args:
        cfg: WorldModelConfig (utilise obs_shape, action_dim, seq_len)
        env_id: ex. "LunarLanderContinuous-v3", "CarRacing-v3"
        n_episodes: nombre d'épisodes à collecter
        max_episode_len: cap par épisode (env reset si done avant)
        policy: "random" | "heuristic"
        seed: graine env + rng
        stride: stride de la sliding window dans une trajectoire continue
    """

    def __init__(
        self,
        cfg: WorldModelConfig,
        env_id: str,
        n_episodes: int,
        max_episode_len: int = 200,
        policy: str = "random",
        seed: int = 0,
        stride: int = 2,
        episode_ids: List[int] | None = None,
    ) -> None:
        import gymnasium as gym

        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.stride = stride
        C, H, W = cfg.obs_shape
        self.target_h = H
        self.target_w = W

        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        env.action_space.seed(seed)
        rng = np.random.default_rng(seed)
        heuristic_fn = HEURISTICS.get(env_id) if policy == "heuristic" else None

        # Stockage : par épisode → liste de frames + actions + dones
        self.episodes: List[dict] = []

        with tqdm(total=n_episodes, desc=f"Collect {env_id} ({policy})") as pbar:
            ep_idx = 0
            while ep_idx < n_episodes:
                obs, _ = env.reset(seed=seed + ep_idx)
                step = 0
                frames: list[np.ndarray] = []  # (H_render, W_render, 3) uint8
                actions: list[np.ndarray] = []
                dones: list[float] = []

                # Capture initial frame
                frame = env.render()
                if frame is None:
                    raise RuntimeError(f"env {env_id} render() returned None — render_mode='rgb_array' required")
                frames.append(frame.astype(np.uint8))

                done = False
                while not done and step < max_episode_len:
                    if heuristic_fn is not None:
                        action = heuristic_fn(step, env, rng)
                    else:
                        action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)

                    next_frame = env.render()
                    frames.append(next_frame.astype(np.uint8))
                    actions.append(_pad_action(action, cfg.action_dim))
                    dones.append(1.0 if done else 0.0)

                    obs = next_obs
                    step += 1

                # Skip ultra-courts épisodes (< seq_len + 1 frames)
                if len(frames) >= self.seq_len + 1:
                    self.episodes.append({
                        "frames": np.stack(frames, axis=0),       # (n+1, Hr, Wr, 3) uint8
                        "actions": np.stack(actions, axis=0),     # (n, action_dim)
                        "dones": np.asarray(dones, dtype=np.float32),
                    })
                    ep_idx += 1
                    pbar.update(1)

        env.close()

        # Filtrage par episode_ids (split train/val)
        if episode_ids is not None:
            keep = set(episode_ids)
            self.episodes = [ep for i, ep in enumerate(self.episodes) if i in keep]

        # Build window index : (ep_idx, start_frame). Exclut fenêtres traversant un done.
        self.windows: List[Tuple[int, int]] = []
        for ei, ep in enumerate(self.episodes):
            n_frames = ep["frames"].shape[0]
            n_actions = ep["actions"].shape[0]
            usable = min(n_frames - 1, n_actions)
            last_start = usable - self.seq_len
            for s in range(0, last_start + 1, self.stride):
                # Exclut si done dans la fenêtre [s, s+T-1]
                if not bool(ep["dones"][s : s + self.seq_len].any()):
                    self.windows.append((ei, s))

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def n_clips(self) -> int:
        """Alias compat VideoClipDataset (= nombre d'épisodes)."""
        return len(self.episodes)

    def __getitem__(self, idx: int):
        ei, start = self.windows[idx]
        ep = self.episodes[ei]
        T = self.seq_len

        # T+1 frames
        obs_seq = torch.empty(T + 1, 3, self.target_h, self.target_w)
        for k in range(T + 1):
            obs_seq[k] = _frame_to_tensor(ep["frames"][start + k], self.target_h, self.target_w)

        action_seq = torch.from_numpy(ep["actions"][start : start + T].astype(np.float32))
        reward_seq = torch.zeros(T)
        done_seq = torch.from_numpy(ep["dones"][start : start + T].astype(np.float32))

        return obs_seq, action_seq, reward_seq, done_seq


def split_episodes(n_episodes: int, val_ratio: float = 0.1, seed: int = 0) -> Tuple[List[int], List[int]]:
    """Split train/val par épisode (PAS par fenêtre, sinon leak)."""
    rng = np.random.default_rng(seed)
    ids = list(range(n_episodes))
    rng.shuffle(ids)
    n_val = max(1, int(n_episodes * val_ratio))
    return ids[n_val:], ids[:n_val]


def make_gym_video_dataloader(
    cfg: WorldModelConfig,
    env_id: str,
    n_episodes: int,
    max_episode_len: int = 200,
    policy: str = "random",
    seed: int = 0,
    stride: int = 2,
    split: str = "train",
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    distributed_sampler: bool = False,
    shared_dataset: GymVideoDataset | None = None,
) -> DataLoader:
    """DataLoader prêt à l'emploi.

    shared_dataset : si fourni, ne re-collecte pas — utile pour partager train/val
    sans doubler le coût de collecte env (qui peut être lent).
    """
    if shared_dataset is None:
        # Collecte 1 fois TOUS les épisodes, on splittera après par filtre
        full_ds = GymVideoDataset(
            cfg, env_id=env_id, n_episodes=n_episodes,
            max_episode_len=max_episode_len, policy=policy,
            seed=seed, stride=stride,
        )
    else:
        full_ds = shared_dataset

    train_ids, val_ids = split_episodes(full_ds.n_clips, val_ratio=val_ratio, seed=seed)
    keep_ids = set(train_ids if split == "train" else val_ids)

    # Filtre les windows en place (sans recopier les épisodes — read-only)
    filtered_windows = [(ei, s) for (ei, s) in full_ds.windows if ei in keep_ids]

    class _Subset(Dataset):
        def __init__(self, base: GymVideoDataset, windows: List[Tuple[int, int]]) -> None:
            self.base = base
            self.windows = windows
            self.n_clips = base.n_clips

        def __len__(self) -> int:
            return len(self.windows)

        def __getitem__(self, idx: int):
            ei, start = self.windows[idx]
            base_ep = self.base.episodes[ei]
            T = self.base.seq_len
            obs_seq = torch.empty(T + 1, 3, self.base.target_h, self.base.target_w)
            for k in range(T + 1):
                obs_seq[k] = _frame_to_tensor(base_ep["frames"][start + k],
                                              self.base.target_h, self.base.target_w)
            action_seq = torch.from_numpy(base_ep["actions"][start : start + T].astype(np.float32))
            reward_seq = torch.zeros(T)
            done_seq = torch.from_numpy(base_ep["dones"][start : start + T].astype(np.float32))
            return obs_seq, action_seq, reward_seq, done_seq

    subset = _Subset(full_ds, filtered_windows)

    sampler = None
    shuffle = (split == "train")
    if distributed_sampler:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(subset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
