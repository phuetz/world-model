"""Probe Wan 2.2 i2v sur 1 clip pour valider le workflow API draft.

Usage : python scripts/dataset_v3/probe_wan22.py [--server 127.0.0.1:8188]
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dataset_v3"))

from comfy_client import ComfyClient
from produce_dataset import _load_workflow
from probe_svd import make_stock_image
from PIL import Image


def patch_wan(wf: dict, prompt: str, image_filename: str, seed: int = 12345,
              out_filename: str = "probe/wan22", width: int = 256, height: int = 256,
              length: int = 25) -> dict:
    wf = json.loads(json.dumps(wf))
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        ins = node.setdefault("inputs", {})
        if ct == "CLIPTextEncode":
            meta_title = (node.get("_meta") or {}).get("title", "").lower()
            if "negative" in meta_title or "neg" in meta_title:
                continue
            ins["text"] = prompt
        if ct == "LoadImage" and "image" in ins:
            ins["image"] = image_filename
        if ct == "WanImageToVideo":
            ins["width"] = width
            ins["height"] = height
            ins["length"] = length
        if ct == "KSamplerAdvanced" and "noise_seed" in ins:
            ins["noise_seed"] = int(seed) & 0x7FFFFFFF
        if ct in ("SaveAnimatedWEBP", "VHS_VideoCombine") and "filename_prefix" in ins:
            ins["filename_prefix"] = out_filename
    return wf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="127.0.0.1:8188")
    p.add_argument("--comfy-input", default="D:/DEV/ComfyUI/input")
    p.add_argument("--prompt", default="a hand placing a wooden cup on a kitchen counter, natural light, photorealistic")
    p.add_argument("--out", default="probe_wan22.webp")
    p.add_argument("--length", type=int, default=25)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    args = p.parse_args()

    img_path = Path(args.comfy_input) / "probe_wan22_source.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    make_stock_image(img_path, size=args.width)
    print(f"stock image -> {img_path}", flush=True)

    client = ComfyClient(f"http://{args.server}")
    if not client.alive():
        print(f"server {args.server} not alive", file=sys.stderr); sys.exit(1)

    wf = _load_workflow(REPO_ROOT / "scripts" / "dataset_v3" / "workflows" / "wan22_i2v.json")
    wf = patch_wan(wf, args.prompt, image_filename="probe_wan22_source.png",
                   width=args.width, height=args.height, length=args.length)

    t0 = time.time()
    pid = client.submit(wf)
    print(f"prompt_id={pid}, waiting up to 900s...", flush=True)
    try:
        outputs = client.wait(pid, timeout=900.0, poll_interval=3.0)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr); sys.exit(2)
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
