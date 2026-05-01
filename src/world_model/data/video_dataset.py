"""Dataset video clips (V3) — Wan 2.2 generated frames pour world-model JEPA.

Lecture lazy depuis disque pour éviter le full-load RAM (1k clips × 120 frames × 256² × 3 ≈ 24 GB
en uint8, ~96 GB en fp32). Pattern miroir de gym_env.SequenceWindowDataset
mais lit les frames PIL à la volée et utilise un optical flow pré-calculé comme action proxy.

Layout disque attendu (généré par scripts/dataset_v3/produce_dataset.py) :

  data/v3_video/<class>/clip_NNNNN/
    frame_000.jpg ... frame_119.jpg          # 120 frames JPEG q=90
    source.png                                 # image SDXL initiale (pas utilisée ici)
    action_proxy.npy                           # shape (119, 4) float32, optical flow Farneback
    meta.json                                  # prompt, classe, seed, gen_time, etc.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ..config.config import WorldModelConfig


def _load_frame(path: str, target_h: int, target_w: int) -> torch.Tensor:
    """JPEG → torch.float32 (C, H, W) normalisé [-1, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    t = torch.from_numpy(arr).float() / 127.5 - 1.0  # [-1, 1]
    t = t.permute(2, 0, 1).unsqueeze(0)              # (1, 3, H, W)
    if t.shape[-2] != target_h or t.shape[-1] != target_w:
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0)


def _scan_clips(root: Path) -> List[Path]:
    """Scan récursif des dossiers clip_NNNNN (peu importe la classe) avec frame_000 + action_proxy."""
    clips: List[Path] = []
    if not root.exists():
        return clips
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for clip_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir() and p.name.startswith("clip_")):
            if (clip_dir / "frame_000.jpg").exists() and (clip_dir / "action_proxy.npy").exists():
                clips.append(clip_dir)
    return clips


class VideoClipDataset(Dataset):
    """Indexe les clips disponibles et expose des fenêtres T+1 frames + T actions.

    Args:
        cfg: WorldModelConfig (utilise obs_shape, action_dim, seq_len)
        root: chemin racine du dataset (e.g. data/v3_video/)
        seq_len: T (longueur de la fenêtre, T+1 frames lues, T actions)
        stride: stride de la sliding window dans un clip
        clip_ids: optionnel, sous-ensemble (split train/val par clip_id)
        blacklist: liste de noms de clip à exclure (e.g. depuis _qa/blacklist.txt)
    """

    def __init__(
        self,
        cfg: WorldModelConfig,
        root: str | Path,
        seq_len: int | None = None,
        stride: int = 4,
        clip_ids: List[str] | None = None,
        blacklist: List[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.seq_len = seq_len if seq_len is not None else cfg.seq_len
        self.stride = stride
        _, self.target_h, self.target_w = cfg.obs_shape

        all_clips = _scan_clips(self.root)
        bl = set(blacklist or [])
        if clip_ids is not None:
            keep = set(clip_ids)
            all_clips = [c for c in all_clips if c.name in keep and c.name not in bl]
        else:
            all_clips = [c for c in all_clips if c.name not in bl]

        self.clips: List[Tuple[Path, int, int]] = []
        # Pré-build l'index plat (clip_dir, n_frames, n_actions)
        for clip_dir in all_clips:
            jpgs = sorted(p for p in clip_dir.iterdir() if p.name.startswith("frame_") and p.suffix == ".jpg")
            n_frames = len(jpgs)
            actions_path = clip_dir / "action_proxy.npy"
            try:
                actions = np.load(actions_path, mmap_mode="r")
            except Exception:
                continue
            n_actions = actions.shape[0]
            usable = min(n_frames - 1, n_actions)
            if usable >= self.seq_len:
                self.clips.append((clip_dir, n_frames, n_actions))

        # Build window index : (clip_idx, start_frame)
        self.windows: List[Tuple[int, int]] = []
        for ci, (_, n_frames, n_actions) in enumerate(self.clips):
            usable = min(n_frames - 1, n_actions)
            last_start = usable - self.seq_len  # start ∈ [0, usable - T]
            for s in range(0, last_start + 1, self.stride):
                self.windows.append((ci, s))

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def n_clips(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        ci, start = self.windows[idx]
        clip_dir, _, _ = self.clips[ci]
        T = self.seq_len

        # T+1 frames
        obs_seq = torch.empty(T + 1, 3, self.target_h, self.target_w)
        for k in range(T + 1):
            frame_path = clip_dir / f"frame_{start + k:03d}.jpg"
            obs_seq[k] = _load_frame(str(frame_path), self.target_h, self.target_w)

        # T actions
        actions_full = np.load(clip_dir / "action_proxy.npy")  # (n_actions, 4)
        action_seq = torch.from_numpy(
            actions_full[start : start + T].astype(np.float32)
        )
        # Pad action_dim si différent
        if action_seq.shape[-1] != self.cfg.action_dim:
            pad = self.cfg.action_dim - action_seq.shape[-1]
            if pad > 0:
                action_seq = torch.cat(
                    [action_seq, torch.zeros(action_seq.shape[0], pad)], dim=-1
                )
            else:
                action_seq = action_seq[:, : self.cfg.action_dim]

        # Reward / done à zéro (dataset vidéo passif)
        reward_seq = torch.zeros(T)
        done_seq = torch.zeros(T)

        return obs_seq, action_seq, reward_seq, done_seq


def split_clips(root: str | Path, val_ratio: float = 0.05, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Split par clip_id (PAS par fenêtre, sinon leak)."""
    clip_paths = _scan_clips(Path(root))
    names = [c.name for c in clip_paths]
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_ratio))
    return names[n_val:], names[:n_val]


def make_video_dataloader(
    cfg: WorldModelConfig,
    root: str | Path,
    seq_len: int | None = None,
    stride: int = 2,
    split: str = "train",
    val_ratio: float = 0.05,
    seed: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    distributed_sampler: bool = False,
    blacklist_path: str | None = None,
) -> DataLoader:
    """DataLoader prêt à l'emploi.

    distributed_sampler=True : pour DDP, retourne le sampler dans loader.sampler
    (pour pouvoir set_epoch).
    """
    train_ids, val_ids = split_clips(root, val_ratio=val_ratio, seed=seed)
    clip_ids = train_ids if split == "train" else val_ids

    blacklist: List[str] = []
    if blacklist_path:
        try:
            with open(blacklist_path, "r", encoding="utf-8") as f:
                blacklist = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        except FileNotFoundError:
            pass

    dataset = VideoClipDataset(
        cfg, root, seq_len=seq_len, stride=stride,
        clip_ids=clip_ids, blacklist=blacklist,
    )
    sampler = None
    shuffle = (split == "train")
    if distributed_sampler:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
