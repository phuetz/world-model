"""Optical flow Farneback comme action proxy 4D pour world-model V3.

Pour chaque paire de frames consécutives (t, t+1), on calcule un flow dense (H, W, 2),
puis on réduit à 4 features compatibles `action_dim=4` :
  [mean_dx, mean_dy, log_magnitude_std, mean_angle]

Le shape de sortie pour un clip de 120 frames est (119, 4) float32.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np


def _load_frame_gray(path: Path) -> np.ndarray:
    """Lazy import cv2 pour ne pas casser l'import du module si OpenCV manque."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def compute_clip_flow(frames: List[Path]) -> np.ndarray:
    """frames : liste ordonnée de chemins de frames JPG/PNG.

    Retourne (T-1, 4) float32 où T = len(frames).
    """
    import cv2
    if len(frames) < 2:
        return np.zeros((0, 4), dtype=np.float32)
    prev = _load_frame_gray(frames[0])
    out = np.zeros((len(frames) - 1, 4), dtype=np.float32)
    for i in range(1, len(frames)):
        cur = _load_frame_gray(frames[i])
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur,
            flow=None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # flow : (H, W, 2), [..., 0] = dx, [..., 1] = dy
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)
        # Clip aux p1-p99 pour réduire l'effet outliers (bordures, occlusions)
        dx_clip = np.clip(dx, np.percentile(dx, 1), np.percentile(dx, 99))
        dy_clip = np.clip(dy, np.percentile(dy, 1), np.percentile(dy, 99))
        out[i - 1, 0] = float(dx_clip.mean())
        out[i - 1, 1] = float(dy_clip.mean())
        # log(1 + std) pour borner et préserver l'info magnitude variability
        out[i - 1, 2] = float(np.log1p(mag.std()))
        out[i - 1, 3] = float(ang.mean())
        prev = cur
    return out


def compute_for_clip_dir(clip_dir: Path) -> Path:
    frames = sorted(p for p in clip_dir.iterdir() if p.suffix == ".jpg" and p.name.startswith("frame_"))
    flow = compute_clip_flow(frames)
    out_path = clip_dir / "action_proxy.npy"
    np.save(out_path, flow.astype(np.float32))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip", type=str, help="single clip dir")
    p.add_argument("--root", type=str, help="root dir with class subfolders")
    p.add_argument("--force", action="store_true", help="recompute even if action_proxy.npy exists")
    args = p.parse_args()

    targets: List[Path] = []
    if args.clip:
        targets.append(Path(args.clip))
    elif args.root:
        root = Path(args.root)
        for cls in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            for clip in sorted(p for p in cls.iterdir() if p.is_dir() and p.name.startswith("clip_")):
                if not args.force and (clip / "action_proxy.npy").exists():
                    continue
                targets.append(clip)
    else:
        print("--clip or --root required", file=sys.stderr)
        sys.exit(2)

    for clip in targets:
        try:
            out = compute_for_clip_dir(clip)
            print(f"OK {out}")
        except Exception as e:
            print(f"FAIL {clip}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
