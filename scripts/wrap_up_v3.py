#!/usr/bin/env python3
"""Wrap-up V3 : enchaîne le training full + eval + plan + summary après la
production overnight. Utilisable au matin (ou auto-trigger quand le producer exit).

Étapes :
  1. Vérifie que le dataset est consistent (>=N clips, blacklist appliquée)
  2. Tue les 2 ComfyUI servers (libère VRAM pour training)
  3. Lance training V3 (50 epochs, single-GPU)
  4. Lance eval_v3.py sur le best checkpoint
  5. Lance plan_v3.py CEM open-loop
  6. Affiche un résumé compact

Usage :
  python scripts/wrap_up_v3.py [--data data/v3_video] [--min-clips 800] [--epochs 50]
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT))
    return proc.returncode


def count_clips(data_root: Path) -> int:
    n = 0
    for cls in data_root.iterdir():
        if not cls.is_dir() or cls.name.startswith("_"):
            continue
        for clip in cls.iterdir():
            if clip.is_dir() and clip.name.startswith("clip_"):
                if (clip / "frame_000.jpg").exists() and (clip / "action_proxy.npy").exists():
                    n += 1
    return n


def kill_comfy_servers() -> None:
    """Stoppe les 2 ComfyUI servers (recherche par CommandLine)."""
    import platform
    if platform.system() != "Windows":
        return
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*ComfyUI*main.py*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(["powershell", "-Command", ps], capture_output=True)
    time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/v3_video")
    p.add_argument("--config", default="configs/v3_video.yaml")
    p.add_argument("--ckpt-dir", default="checkpoints_v3_video")
    p.add_argument("--log-dir", default="runs/v3_video")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--min-clips", type=int, default=800)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-kill", action="store_true", help="ne tue pas ComfyUI")
    p.add_argument("--skip-qa", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-plan", action="store_true")
    args = p.parse_args()

    data = Path(args.data)
    if not data.exists():
        print(f"FATAL : data root {data} absent", file=sys.stderr)
        sys.exit(2)

    n = count_clips(data)
    print(f"[wrap-up] {n} clips presents dans {data}")
    if n < args.min_clips:
        print(f"WARN : {n} < {args.min_clips} (continue quand meme)", file=sys.stderr)

    # 1) QA + blacklist
    if not args.skip_qa:
        run([sys.executable, "scripts/dataset_v3/qa_dataset.py", "--root", str(data)])

    # 2) Tue les ComfyUI servers
    if not args.skip_kill:
        print("[wrap-up] kill ComfyUI servers...")
        kill_comfy_servers()

    # 3) Training V3 full
    if not args.skip_train:
        # Override max_epochs via env (si on veut moins). Pour l'instant on prend la config.
        train_cmd = [
            sys.executable, "scripts/train_v3.py",
            "--config", args.config,
            "--data", str(data),
            "--gpus", "1",
            "--num-workers", str(args.num_workers),
            "--ckpt-dir", args.ckpt_dir,
            "--log-dir", args.log_dir,
        ]
        bl = data / "_qa" / "blacklist.txt"
        if bl.exists():
            train_cmd += ["--blacklist", str(bl)]
        rc = run(train_cmd)
        if rc != 0:
            print(f"WARN : training exit {rc}", file=sys.stderr)

    # 4) Eval
    ckpts = sorted(Path(args.ckpt_dir).glob("epoch_*.pt"))
    best_ckpt = ckpts[-1] if ckpts else None

    if not args.skip_eval and best_ckpt:
        eval_cmd = [
            sys.executable, "scripts/eval_v3.py",
            "--checkpoint", str(best_ckpt),
            "--config", args.config,
            "--data", str(data),
            "--horizons", "1,2,4,8,16",
            "--max-windows", "2000",
            "--report", "eval_report_v3_video.md",
        ]
        run(eval_cmd)

    # 5) Plan CEM
    if not args.skip_plan and best_ckpt:
        plan_cmd = [
            sys.executable, "scripts/plan_v3.py",
            "--checkpoint", str(best_ckpt),
            "--config", args.config,
            "--data", str(data),
            "--n-pairs", "100",
            "--horizon", "8",
            "--report", "plan_report_v3.md",
        ]
        run(plan_cmd)

    # 6) Summary
    print("\n=== WRAP-UP V3 SUMMARY ===")
    print(f"clips:        {n}")
    print(f"checkpoints:  {len(ckpts)} ({best_ckpt.name if best_ckpt else 'none'})")
    for report in ("eval_report_v3_video.md", "plan_report_v3.md"):
        rp = REPO_ROOT / report
        if rp.exists():
            print(f"\n--- {report} (premières lignes) ---")
            with open(rp, "r", encoding="utf-8") as f:
                lines = f.readlines()[:30]
            print("".join(lines))


if __name__ == "__main__":
    main()
