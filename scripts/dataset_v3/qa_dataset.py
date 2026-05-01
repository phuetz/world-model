"""QA du dataset video : contact sheet + stats + blacklist auto.

Outputs dans data/v3_video/_qa/ :
  - contact_sheet.png    : grid 5x6 (frame_0 + frame_mid + frame_last) × 6 clips
  - stats.json           : count par classe, mean optical flow magnitude, % dégénéré
  - flow_histogram.png   : distribution mean_dx/dy par classe
  - blacklist.txt        : liste des clip_ids à exclure (un par ligne)

Usage :
  python scripts/dataset_v3/qa_dataset.py --root data/v3_video
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


def load_meta(clip_dir: Path) -> dict:
    try:
        with open(clip_dir / "meta.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def gather(root: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    all_clips: list[Path] = []
    by_class: dict[str, list[Path]] = defaultdict(list)
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for clip in sorted(p for p in cls_dir.iterdir() if p.is_dir() and p.name.startswith("clip_")):
            if (clip / "frame_000.jpg").exists() and (clip / "action_proxy.npy").exists():
                all_clips.append(clip)
                by_class[cls_dir.name].append(clip)
    return all_clips, by_class


def build_contact_sheet(clips: list[Path], out: Path, n_clips: int = 6, size: int = 192) -> None:
    sample = clips[: n_clips]
    cols = 3
    rows = max(1, len(sample))
    grid = Image.new("RGB", (cols * size, rows * size), (32, 32, 32))
    for r, clip in enumerate(sample):
        frames = sorted(clip.glob("frame_*.jpg"))
        if len(frames) < 3:
            continue
        triple = [frames[0], frames[len(frames) // 2], frames[-1]]
        for c, fp in enumerate(triple):
            with Image.open(fp) as im:
                im = im.convert("RGB").resize((size, size), Image.BILINEAR)
                grid.paste(im, (c * size, r * size))
    grid.save(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    qa_dir = root / "_qa"
    qa_dir.mkdir(exist_ok=True)

    clips, by_class = gather(root)
    print(f"clips total: {len(clips)} | classes: {len(by_class)}", flush=True)

    flow_means: dict[str, list[float]] = defaultdict(list)
    flow_p99: dict[str, list[float]] = defaultdict(list)
    deg_clips: list[str] = []
    deg_reasons: dict[str, str] = {}
    for clip in clips:
        meta = load_meta(clip)
        cls = meta.get("class") or clip.parent.name
        try:
            flow = np.load(clip / "action_proxy.npy")
        except Exception:
            deg_clips.append(clip.name); deg_reasons[clip.name] = "action_proxy.npy unreadable"
            continue
        if flow.size == 0:
            deg_clips.append(clip.name); deg_reasons[clip.name] = "action_proxy empty"
            continue
        mag = np.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        m = float(mag.mean())
        p99 = float(np.percentile(mag, 99))
        flow_means[cls].append(m)
        flow_p99[cls].append(p99)
        # Critères de dégénérescence
        # 1) flow magnitude p99 < 0.05 → caméra fixe + objet immobile
        if p99 < 0.05:
            deg_clips.append(clip.name); deg_reasons[clip.name] = f"flow p99 {p99:.3f} < 0.05"
            continue
        # 2) frame count != attendu (clip tronqué)
        frames = sorted(clip.glob("frame_*.jpg"))
        n_frames = len(frames)
        expected = meta.get("n_frames", 0)
        if expected and n_frames < int(0.9 * expected):
            deg_clips.append(clip.name); deg_reasons[clip.name] = f"only {n_frames}/{expected} frames"
            continue
        # 3) Frames quasi-identiques : pixel variance entre frame 0 et frame mid
        if n_frames >= 3:
            try:
                arr0 = np.asarray(Image.open(frames[0]).convert("RGB"), dtype=np.float32)
                arrM = np.asarray(Image.open(frames[len(frames) // 2]).convert("RGB"), dtype=np.float32)
                pixel_diff = float(np.abs(arr0 - arrM).mean())
                if pixel_diff < 1.5:  # JPEG q=90 -> bruit ~0.5-1, real motion >> 5
                    deg_clips.append(clip.name); deg_reasons[clip.name] = f"frames identiques (mean diff {pixel_diff:.2f})"
                    continue
            except Exception:
                pass

    stats = {
        "total_clips": len(clips),
        "classes": {
            cls: {
                "count": len(by_class[cls]),
                "flow_mean_mean": float(np.mean(flow_means[cls])) if flow_means[cls] else None,
                "flow_mean_std": float(np.std(flow_means[cls])) if flow_means[cls] else None,
                "flow_p99_mean": float(np.mean(flow_p99[cls])) if flow_p99[cls] else None,
            }
            for cls in by_class
        },
        "n_blacklisted": len(deg_clips),
    }
    with open(qa_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(qa_dir / "blacklist.txt", "w", encoding="utf-8") as f:
        for c in deg_clips:
            reason = deg_reasons.get(c, "?")
            f.write(f"{c}  # {reason}\n")

    build_contact_sheet(clips, qa_dir / "contact_sheet.png", n_clips=6, size=192)
    print(f"stats: {qa_dir/'stats.json'}", flush=True)
    print(f"contact sheet: {qa_dir/'contact_sheet.png'}", flush=True)
    print(f"blacklisted: {len(deg_clips)} clips", flush=True)


if __name__ == "__main__":
    main()
