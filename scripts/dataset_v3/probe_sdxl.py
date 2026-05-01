"""Probe SDXL : valide que ComfyUI server répond + Juggernaut-XL génère une image.

Usage : python scripts/dataset_v3/probe_sdxl.py [--server 127.0.0.1:8188]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dataset_v3"))

from comfy_client import ComfyClient
from produce_dataset import _load_workflow, patch_sdxl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="127.0.0.1:8188")
    p.add_argument("--prompt", default="a cozy kitchen counter with a wooden cutting board, sunlit, photorealistic")
    p.add_argument("--out", default="probe_sdxl.png")
    args = p.parse_args()

    client = ComfyClient(f"http://{args.server}")
    if not client.alive():
        print(f"server {args.server} not alive", file=sys.stderr)
        sys.exit(1)

    wf = _load_workflow(REPO_ROOT / "scripts" / "dataset_v3" / "workflows" / "sdxl_image.json")
    wf = patch_sdxl(wf, args.prompt, seed=12345, out_filename="probe/sdxl")

    t0 = time.time()
    print(f"submit prompt to {args.server}", flush=True)
    pid = client.submit(wf)
    print(f"prompt_id={pid}, polling /history...", flush=True)
    outputs = client.wait(pid, timeout=180.0)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s", flush=True)

    images = client.collect_images(outputs)
    if not images:
        print("FAIL : no images returned", file=sys.stderr)
        sys.exit(2)
    fn, data = images[0]
    out_path = Path(args.out)
    out_path.write_bytes(data)
    print(f"saved {len(data)/1024:.0f} KB to {out_path} (filename: {fn})")


if __name__ == "__main__":
    main()
