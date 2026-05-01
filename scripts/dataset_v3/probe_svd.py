"""Probe SVD-XT : valide gen vidéo i2v sur image stock (PIL noise/gradient).

Test directement SVD-XT sans dépendre d'un générateur d'image source en amont.
Permet de valider la moitié vidéo du pipeline pendant que sd_turbo est en download.

Usage : python scripts/dataset_v3/probe_svd.py [--server 127.0.0.1:8188]
"""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dataset_v3"))

from comfy_client import ComfyClient
from produce_dataset import _load_workflow


def make_stock_image(out: Path, size: int = 256) -> None:
    """Image source : gradient + noise pour avoir une structure non-triviale."""
    rng = np.random.default_rng(0)
    base = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.stack([
        np.broadcast_to(base[:, None], (size, size)).astype(np.uint8),  # R gradient horizontal
        np.broadcast_to(base[None, :], (size, size)).astype(np.uint8),  # G gradient vertical
        np.full((size, size), 128, dtype=np.uint8),                      # B constant
    ], axis=-1)
    noise = rng.integers(-30, 30, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out)


def patch_svd(wf: dict, image_filename: str, seed: int = 12345, out_filename: str = "probe/svd",
              width: int = 256, height: int = 256, frames: int = 25) -> dict:
    wf = json.loads(json.dumps(wf))
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        ins = node.setdefault("inputs", {})
        if ct == "LoadImage" and "image" in ins:
            ins["image"] = image_filename
        if ct == "SVD_img2vid_Conditioning":
            ins["width"] = width
            ins["height"] = height
            ins["video_frames"] = frames
        if ct == "KSampler" and "seed" in ins:
            ins["seed"] = int(seed) & 0x7FFFFFFF
        if ct == "SaveAnimatedWEBP" and "filename_prefix" in ins:
            ins["filename_prefix"] = out_filename
    return wf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="127.0.0.1:8188")
    p.add_argument("--comfy-input", default="D:/DEV/ComfyUI/input")
    p.add_argument("--frames", type=int, default=25)
    p.add_argument("--out", default="probe_svd.webp")
    args = p.parse_args()

    img_path = Path(args.comfy_input) / "probe_svd_source.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    make_stock_image(img_path)
    print(f"stock image written to {img_path}", flush=True)

    client = ComfyClient(f"http://{args.server}")
    if not client.alive():
        print(f"server {args.server} not alive", file=sys.stderr)
        sys.exit(1)
    wf = _load_workflow(REPO_ROOT / "scripts" / "dataset_v3" / "workflows" / "svd_i2v.json")
    wf = patch_svd(wf, image_filename="probe_svd_source.png", frames=args.frames)

    t0 = time.time()
    pid = client.submit(wf)
    print(f"prompt_id={pid}, polling /history (timeout 600s)...", flush=True)
    outputs = client.wait(pid, timeout=600.0, poll_interval=2.0)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s", flush=True)

    images = client.collect_images(outputs)
    if not images:
        print("FAIL : no output", file=sys.stderr); sys.exit(2)
    fn, data = images[0]
    Path(args.out).write_bytes(data)
    print(f"saved {len(data)/1024:.0f} KB to {args.out} (filename: {fn})")


if __name__ == "__main__":
    main()
