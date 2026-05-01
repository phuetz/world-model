"""Producer overnight : lit prompts.jsonl, dispatche sur N serveurs ComfyUI,
écrit les frames + optical flow + meta dans data/v3_video/<class>/<clip_id>/.

Resumable : skip les clip_id présents dans progress.jsonl.
Watchdog disque + watchdog server timeout.

Usage :
  python produce_dataset.py --prompts scripts/dataset_v3/prompts.jsonl \\
      --servers 127.0.0.1:8188,127.0.0.1:8189 --out data/v3_video --target 1500
"""
from __future__ import annotations
import argparse
import io
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dataset_v3"))

import psutil
from PIL import Image

from comfy_client import ComfyClient
from optical_flow import compute_clip_flow
from stock_image import make_for_class


# --- Workflow loading & patching -------------------------------------------

def _load_workflow(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _patch_node_value(workflow: Dict[str, Any], node_id_or_title: str, key_path: List[str], value: Any) -> None:
    """Mutate workflow inputs[key_path] for the matching node (by id or _meta.title)."""
    nodes = workflow if isinstance(workflow, dict) else {}
    target = None
    if node_id_or_title in nodes:
        target = nodes[node_id_or_title]
    else:
        for nid, node in nodes.items():
            meta = node.get("_meta") or {}
            if meta.get("title") == node_id_or_title:
                target = node
                break
    if target is None:
        return
    inputs = target.setdefault("inputs", {})
    obj = inputs
    for k in key_path[:-1]:
        obj = obj.setdefault(k, {})
    obj[key_path[-1]] = value


def patch_sdxl(workflow: Dict[str, Any], prompt: str, seed: int, out_filename: str) -> Dict[str, Any]:
    """Patch d'un workflow SDXL standard (KSampler-based)."""
    wf = json.loads(json.dumps(workflow))  # deep copy
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        ins = node.setdefault("inputs", {})
        if ct == "CLIPTextEncode" and "text" in ins:
            # Le premier (positive) reçoit notre prompt ; on laisse le négatif par défaut
            meta_title = (node.get("_meta") or {}).get("title", "").lower()
            if "negative" in meta_title or "neg" in meta_title:
                continue
            ins["text"] = prompt
        if ct == "KSampler" and "seed" in ins:
            ins["seed"] = int(seed) & 0x7FFFFFFF
        if ct == "SaveImage" and "filename_prefix" in ins:
            ins["filename_prefix"] = out_filename
    return wf


def patch_wan_i2v(
    workflow: Dict[str, Any],
    prompt: str,
    image_filename: str,
    seed: int,
    out_filename: str,
    width: int = 256,
    height: int = 256,
    length: int = 121,
) -> Dict[str, Any]:
    """Patch d'un workflow Wan 2.2 i2v."""
    wf = json.loads(json.dumps(workflow))
    for nid, node in wf.items():
        ct = node.get("class_type", "")
        ins = node.setdefault("inputs", {})
        if ct == "CLIPTextEncode" and "text" in ins:
            meta_title = (node.get("_meta") or {}).get("title", "").lower()
            if "negative" in meta_title or "neg" in meta_title:
                continue
            ins["text"] = prompt
        if ct == "LoadImage" and "image" in ins:
            ins["image"] = image_filename
        if ct == "KSampler" and "seed" in ins:
            ins["seed"] = int(seed) & 0x7FFFFFFF
        if ct in ("EmptyLatentImage", "EmptyHunyuanLatentVideo", "WanImageToVideo") and "width" in ins:
            ins["width"] = width
            if "height" in ins:
                ins["height"] = height
            if "length" in ins:
                ins["length"] = length
        if ct in ("SaveImage", "SaveAnimatedWEBP", "VHS_VideoCombine"):
            if "filename_prefix" in ins:
                ins["filename_prefix"] = out_filename
    return wf


# --- Worker pipeline --------------------------------------------------------

class Producer:
    def __init__(
        self,
        prompts_path: Path,
        servers: List[str],
        out_root: Path,
        sdxl_workflow_path: Path,
        wan_workflow_path: Path,
        comfy_input_dir: Path,
        target: int,
        min_disk_gb: float = 50.0,
        clip_length: int = 120,
        size: int = 256,
    ) -> None:
        self.out_root = out_root
        self.servers = servers
        self.target = target
        self.min_disk_gb = min_disk_gb
        self.clip_length = clip_length
        self.size = size
        self.comfy_input_dir = comfy_input_dir

        self.sdxl_wf = _load_workflow(sdxl_workflow_path)
        self.wan_wf = _load_workflow(wan_workflow_path)

        import random
        self.prompts: List[Dict[str, Any]] = []
        with open(prompts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.prompts.append(json.loads(line))
        # Shuffle déterministe (seed fixe) — équilibre les classes au sein du
        # dataset même si la production est interrompue avant la fin.
        random.Random(42).shuffle(self.prompts)
        self.progress_path = out_root / "progress.jsonl"
        self.lock = threading.Lock()
        self.done_ids: set[str] = self._load_done()
        self.work_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        for p in self.prompts[: self.target]:
            if p["id"] not in self.done_ids:
                self.work_queue.put(p)
        self.stop_evt = threading.Event()

    def _load_done(self) -> set[str]:
        done: set[str] = set()
        if self.progress_path.exists():
            with open(self.progress_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if obj.get("status") == "ok":
                            done.add(obj["clip_id"])
                    except Exception:
                        pass
        return done

    def _log_progress(self, obj: Dict[str, Any]) -> None:
        with self.lock:
            with open(self.progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")

    def _check_disk(self) -> bool:
        anchor = self.out_root.resolve().anchor or os.getcwd()
        free_gb = psutil.disk_usage(str(anchor)).free / (1024 ** 3)
        return free_gb >= self.min_disk_gb

    def _process_one(self, client: ComfyClient, item: Dict[str, Any]) -> Dict[str, Any]:
        cls = item["class"]
        clip_id = item["id"]
        seed = item["seed"]
        clip_dir = self.out_root / cls / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        # ---- 1) Source image (procedural stock — pas de SDXL nécessaire) ---
        src_img = make_for_class(cls, seed, self.size)
        src_img.save(clip_dir / "source.png")
        input_filename = f"v3src_{clip_id}.png"
        input_path = self.comfy_input_dir / input_filename
        src_img.save(input_path)

        # ---- 2) SVD-XT i2v -------------------------------------------------
        from probe_svd import patch_svd
        wan_wf = patch_svd(
            self.wan_wf, image_filename=input_filename,
            seed=seed, out_filename=f"v3svd/{clip_id}",
            width=self.size, height=self.size, frames=self.clip_length,
        )
        prompt_id = client.submit(wan_wf)
        outputs = client.wait(prompt_id, timeout=900.0)
        # Récupère la vidéo / les frames
        files = client.collect_images(outputs)
        # On essaie de gérer plusieurs formats : webp/mp4 unique, ou bien frames
        frames: List[bytes] = []
        for fn, data in files:
            ext = Path(fn).suffix.lower()
            if ext in (".png", ".jpg", ".jpeg"):
                frames.append(data)
            elif ext in (".webp", ".mp4", ".webm", ".gif"):
                # Décompose via PIL (webp/gif) ou ffmpeg (mp4)
                frames.extend(self._extract_frames(data, ext))
        # Tronque à clip_length
        if len(frames) < 2:
            raise RuntimeError(f"wan produced {len(frames)} frames for {clip_id}")
        frames = frames[: self.clip_length]
        # Save frames JPG q=90
        from PIL import Image as PImage
        for k, b in enumerate(frames):
            with PImage.open(io.BytesIO(b)) as im:
                im = im.convert("RGB")
                if im.size != (self.size, self.size):
                    im = im.resize((self.size, self.size), PImage.BILINEAR)
                im.save(clip_dir / f"frame_{k:03d}.jpg", quality=90)

        # ---- 3) Optical flow -----------------------------------------------
        frame_paths = sorted(clip_dir.glob("frame_*.jpg"))
        flow = compute_clip_flow(frame_paths)
        import numpy as np
        np.save(clip_dir / "action_proxy.npy", flow.astype(np.float32))

        # ---- 4) Meta -------------------------------------------------------
        gen_time = time.time() - t0
        meta = {
            "clip_id": clip_id,
            "class": cls,
            "prompt": item["prompt"],
            "source_image_prompt": item["source_image_prompt"],
            "seed": seed,
            "n_frames": len(frame_paths),
            "n_actions": int(flow.shape[0]),
            "size": self.size,
            "gen_time_s": gen_time,
        }
        with open(clip_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Cleanup comfy input file
        try:
            input_path.unlink()
        except FileNotFoundError:
            pass
        return meta

    def _extract_frames(self, data: bytes, ext: str) -> List[bytes]:
        """webp/gif via PIL, mp4/webm via imageio si dispo, sinon ffmpeg subprocess."""
        from PIL import Image as PImage
        out: List[bytes] = []
        if ext in (".webp", ".gif"):
            with PImage.open(io.BytesIO(data)) as im:
                try:
                    n = im.n_frames
                except AttributeError:
                    n = 1
                for k in range(n):
                    im.seek(k)
                    rgb = im.convert("RGB")
                    buf = io.BytesIO()
                    rgb.save(buf, format="JPEG", quality=90)
                    out.append(buf.getvalue())
            return out
        # mp4/webm via imageio
        try:
            import imageio.v3 as iio  # type: ignore
            arr = iio.imread(io.BytesIO(data), index=None)  # (T, H, W, C)
            for frame in arr:
                im = PImage.fromarray(frame).convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=90)
                out.append(buf.getvalue())
            return out
        except Exception:
            pass
        # ffmpeg fallback
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            inp = tdp / f"in{ext}"
            inp.write_bytes(data)
            outdir = tdp / "frames"
            outdir.mkdir()
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(inp),
                    "-vsync", "0", "-q:v", "2",
                    str(outdir / "f_%05d.jpg"),
                ],
                check=False, capture_output=True,
            )
            for fp in sorted(outdir.glob("*.jpg")):
                out.append(fp.read_bytes())
        return out

    def worker(self, server: str) -> None:
        client = ComfyClient(f"http://{server}")
        if not client.alive():
            print(f"[worker {server}] not alive, skipping", flush=True)
            return
        print(f"[worker {server}] started", flush=True)
        while not self.stop_evt.is_set():
            try:
                item = self.work_queue.get(timeout=2.0)
            except queue.Empty:
                return
            if not self._check_disk():
                print(f"[worker {server}] disk low, stopping", flush=True)
                self.stop_evt.set()
                return
            t0 = time.time()
            try:
                meta = self._process_one(client, item)
                self._log_progress({"clip_id": item["id"], "status": "ok", "server": server, "duration": time.time() - t0})
                print(f"[worker {server}] OK {item['id']} ({meta['gen_time_s']:.1f}s)", flush=True)
            except Exception as e:
                tb = traceback.format_exc()
                self._log_progress({"clip_id": item["id"], "status": "fail", "error": str(e)[:200], "server": server})
                print(f"[worker {server}] FAIL {item['id']}: {e}\n{tb}", flush=True)

    def run(self) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        threads = []
        for s in self.servers:
            t = threading.Thread(target=self.worker, args=(s,), daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("[producer] all workers done")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", required=True)
    p.add_argument("--servers", required=True, help="comma-separated host:port")
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=1500)
    p.add_argument("--sdxl-workflow", default="scripts/dataset_v3/workflows/sdxl_image.json")
    p.add_argument("--wan-workflow", default="scripts/dataset_v3/workflows/svd_i2v.json")
    p.add_argument("--comfy-input", default="D:/DEV/ComfyUI/input")
    p.add_argument("--min-disk-gb", type=float, default=50.0)
    p.add_argument("--clip-length", type=int, default=120)
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    prod = Producer(
        prompts_path=Path(args.prompts),
        servers=[s.strip() for s in args.servers.split(",")],
        out_root=Path(args.out),
        sdxl_workflow_path=Path(args.sdxl_workflow),
        wan_workflow_path=Path(args.wan_workflow),
        comfy_input_dir=Path(args.comfy_input),
        target=args.target,
        min_disk_gb=args.min_disk_gb,
        clip_length=args.clip_length,
        size=args.size,
    )
    prod.run()


if __name__ == "__main__":
    main()
