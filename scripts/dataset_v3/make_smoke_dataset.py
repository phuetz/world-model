"""Génère un mini dataset synthétique au format VideoClipDataset pour dry-run DDP.

Crée N clips de 32 frames 256×256 (bruit gaussien spatialement lissé pour simuler
des frames + actions aléatoires) afin de valider le pipeline DDP avant la production.

Usage : python scripts/dataset_v3/make_smoke_dataset.py --out data/v3_video_smoke --clips 50
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def make_smoke(out_root: Path, n_clips: int = 50, frames_per_clip: int = 32, size: int = 256, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    cls_dir = out_root / "smoke"
    cls_dir.mkdir(parents=True, exist_ok=True)
    for ci in range(n_clips):
        clip_dir = cls_dir / f"clip_{ci:05d}"
        clip_dir.mkdir(exist_ok=True)
        # Frame de base + drift temporel pour simuler de la dynamique
        base = rng.normal(0.5, 0.15, size=(size, size, 3)).clip(0, 1)
        drift = rng.normal(0.0, 0.005, size=(size, size, 3))
        for k in range(frames_per_clip):
            arr = np.clip(base + drift * k, 0, 1)
            arr_u8 = (arr * 255).astype(np.uint8)
            Image.fromarray(arr_u8).save(clip_dir / f"frame_{k:03d}.jpg", quality=85)
        # action_proxy : (frames_per_clip - 1, 4) — drift mean encoded
        actions = rng.normal(0.0, 0.5, size=(frames_per_clip - 1, 4)).astype(np.float32)
        np.save(clip_dir / "action_proxy.npy", actions)
        Image.fromarray(arr_u8).save(clip_dir / "source.png")
        with open(clip_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"clip_id": clip_dir.name, "class": "smoke", "smoke": True, "n_frames": frames_per_clip}, f)
    print(f"smoke dataset OK : {n_clips} clips × {frames_per_clip} frames @ {out_root}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/v3_video_smoke")
    p.add_argument("--clips", type=int, default=50)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()
    make_smoke(Path(args.out), n_clips=args.clips, frames_per_clip=args.frames, size=args.size)


if __name__ == "__main__":
    main()
