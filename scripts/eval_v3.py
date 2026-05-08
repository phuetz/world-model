#!/usr/bin/env python3
"""Évaluation V3 : MSE@horizons + effective rank + variance per-dim sur dataset video.

Usage :
  python scripts/eval_v3.py \\
      --checkpoint checkpoints_v3_video/epoch_0050.pt \\
      --config configs/v3_video.yaml \\
      --data data/v3_video \\
      --horizons 1,2,4,8,16
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from world_model.config.config import WorldModelConfig  # noqa: E402
from world_model.data.video_dataset import (  # noqa: E402
    VideoClipDataset, split_clips,
)
from world_model.data.gym_video_dataset import (  # noqa: E402
    GymVideoDataset, split_episodes,
)
from world_model.models.world_model import WorldModel  # noqa: E402

# Réutilise les helpers existants
sys.path.insert(0, str(REPO_ROOT))
from scripts.eval import (  # type: ignore  # noqa: E402
    effective_rank, format_float, strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/v3_video.yaml")
    p.add_argument("--backend", default="video", choices=["video", "gym"],
                   help="video = VideoClipDataset (V3), gym = GymVideoDataset (V4)")
    p.add_argument("--data", default="data/v3_video", help="root dataset (backend=video)")
    # backend=gym
    p.add_argument("--env", default="LunarLanderContinuous-v3")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--max-episode-len", type=int, default=200)
    p.add_argument("--policy", default="heuristic", choices=["random", "heuristic"])
    p.add_argument("--horizons", default="1,2,4,8,16")
    p.add_argument("--max-windows", type=int, default=2000, help="cap pour eval rapide")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default="eval_report_v3_video.md")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def build_val_set(args: argparse.Namespace, cfg: WorldModelConfig):
    """Construit le val set selon backend. Retourne un Dataset compatible :
    val_set[i] -> (obs_seq, action_seq, reward_seq, done_seq), len(val_set), val_set.n_clips."""
    if args.backend == "video":
        train_ids, val_ids = split_clips(args.data, val_ratio=0.05, seed=args.seed)
        return VideoClipDataset(cfg, args.data, seq_len=cfg.seq_len, stride=2, clip_ids=val_ids)
    # gym
    full = GymVideoDataset(
        cfg, env_id=args.env, n_episodes=args.n_episodes,
        max_episode_len=args.max_episode_len, policy=args.policy,
        seed=args.seed, stride=2,
    )
    train_ids, val_ids = split_episodes(full.n_clips, val_ratio=0.1, seed=args.seed)
    keep = set(val_ids)

    class _Subset:
        def __init__(self, base: GymVideoDataset, windows):
            self.base = base
            self.windows = windows
            self.n_clips = base.n_clips
        def __len__(self):
            return len(self.windows)
        def __getitem__(self, idx):
            ei, start = self.windows[idx]
            base_ep = self.base.episodes[ei]
            T = self.base.seq_len
            obs_seq = torch.empty(T + 1, 3, self.base.target_h, self.base.target_w)
            from world_model.data.gym_video_dataset import _frame_to_tensor
            for k in range(T + 1):
                obs_seq[k] = _frame_to_tensor(base_ep["frames"][start + k],
                                              self.base.target_h, self.base.target_w)
            action_seq = torch.from_numpy(base_ep["actions"][start : start + T].astype("float32"))
            reward_seq = torch.zeros(T)
            done_seq = torch.from_numpy(base_ep["dones"][start : start + T].astype("float32"))
            return obs_seq, action_seq, reward_seq, done_seq

    filtered = [(ei, s) for (ei, s) in full.windows if ei in keep]
    return _Subset(full, filtered)


def load_model(cfg: WorldModelConfig, ckpt_path: str, device: torch.device) -> WorldModel:
    model = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        state = ckpt
    model.load_state_dict(strip_module_prefix(state))
    model.eval()
    return model


def horizon_mse(
    model: WorldModel,
    val_set,
    horizon: int,
    device: torch.device,
    batch_size: int,
    max_windows: int,
) -> Tuple[float, int]:
    """Pour chaque fenêtre, encode obs[0], rollout autoregressif sur `horizon` steps,
    compare au z_target = encode(obs[horizon])."""
    n = min(len(val_set), max_windows)
    if n == 0:
        return float("nan"), 0
    losses: List[torch.Tensor] = []
    with torch.no_grad():
        for offset in range(0, n, batch_size):
            batch = [val_set[i] for i in range(offset, min(offset + batch_size, n))]
            obs_seq = torch.stack([b[0] for b in batch], dim=0).to(device)
            action_seq = torch.stack([b[1] for b in batch], dim=0).to(device)
            B, Tp1 = obs_seq.shape[0], obs_seq.shape[1]
            if horizon >= Tp1:
                continue
            z = model.encode(obs_seq[:, 0])
            for step in range(horizon):
                z = model.predict_next(z, action_seq[:, step])
            z_target = model.encode(obs_seq[:, horizon])
            mse_per_sample = F.mse_loss(z, z_target, reduction="none").mean(dim=1)
            losses.append(mse_per_sample.detach().cpu())
    if not losses:
        return float("nan"), 0
    cat = torch.cat(losses)
    return cat.mean().item(), int(cat.numel())


def collect_latents(
    model: WorldModel,
    val_set,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> torch.Tensor:
    out: List[torch.Tensor] = []
    n = min(len(val_set), max_samples)
    with torch.no_grad():
        for offset in range(0, n, batch_size):
            batch = [val_set[i] for i in range(offset, min(offset + batch_size, n))]
            obs0 = torch.stack([b[0][0] for b in batch], dim=0).to(device)
            z = model.encode(obs0)
            out.append(z.detach().cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0)


def build_report(
    checkpoint: str,
    config_path: str,
    data_root: str,
    device: torch.device,
    mse_by_horizon: Dict[int, Tuple[float, int]],
    latents: torch.Tensor,
    latent_dim: int,
) -> str:
    var_per_dim = latents.float().var(dim=0, unbiased=False) if latents.numel() > 0 else torch.empty(0)
    var_mean = float(var_per_dim.mean()) if var_per_dim.numel() > 0 else float("nan")
    var_min = float(var_per_dim.min()) if var_per_dim.numel() > 0 else float("nan")
    var_max = float(var_per_dim.max()) if var_per_dim.numel() > 0 else float("nan")
    rank_eff = effective_rank(latents) if latents.numel() > 0 else 0.0
    rank_pct = 100.0 * rank_eff / latent_dim if latent_dim > 0 else 0.0

    h1_mse = mse_by_horizon.get(1, (float("nan"), 0))[0]
    hN = max(mse_by_horizon.keys()) if mse_by_horizon else 1
    hN_mse = mse_by_horizon.get(hN, (float("nan"), 0))[0]
    compounding = (hN_mse / h1_mse) if (h1_mse and h1_mse == h1_mse and h1_mse > 0) else float("nan")

    rows = ["| Horizon | MSE latent | Windows |", "|---:|---:|---:|"]
    for h in sorted(mse_by_horizon.keys()):
        mse, n = mse_by_horizon[h]
        rows.append(f"| {h} | {format_float(mse)} | {n} |")

    lines = [
        "# Rapport d'évaluation V3 (video)",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Config: `{config_path}`",
        f"- Dataset: `{data_root}`",
        f"- Device: `{device}`",
        f"- Date: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## MSE multi-step",
        "",
        *rows,
        "",
        f"- **Compounding ratio MSE(h={hN}) / MSE(h=1)** : `{format_float(compounding)}`",
        "",
        "## Statistiques latents",
        "",
        f"- Variance moyenne / dim : `{var_mean:.6f}`",
        f"- Variance min / dim : `{var_min:.6f}`",
        f"- Variance max / dim : `{var_max:.6f}`",
        f"- **Effective rank** : `{rank_eff:.2f}` / `{latent_dim}` (`{rank_pct:.2f}%`)",
        "",
        "## Comparaison V1.8 baseline (CarRacing)",
        "",
        "| Métrique | V1.8 | V3 |",
        "|---|---|---|",
        f"| MSE h=1 | 0.0135 | {format_float(h1_mse)} |",
        f"| Compounding (V1.8 h=20, V3 h={hN}) | x2.8 | {format_float(compounding)} |",
        f"| Effective rank | 20.6/256 (8%) | {rank_eff:.1f}/{latent_dim} ({rank_pct:.1f}%) |",
        "",
        "## Interprétation",
        "",
        (
            f"Rank {rank_eff:.1f}/{latent_dim} → le modèle utilise {rank_pct:.1f}% "
            f"de sa capacité latente. Cible >15% = succès architectural ; "
            f"si <10% → re-tune lambda_var en V3.0.1."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = WorldModelConfig.from_yaml(args.config)
    model = load_model(cfg, args.checkpoint, device)

    val_set = build_val_set(args, cfg)
    src_label = args.data if args.backend == "video" else f"{args.env} ({args.policy}, {args.n_episodes} ep)"
    print(f"[eval_v3] backend={args.backend} val_clips={val_set.n_clips} val_windows={len(val_set)} horizons={horizons}", flush=True)

    mse_by_horizon: Dict[int, Tuple[float, int]] = {}
    for h in horizons:
        mse, n = horizon_mse(model, val_set, h, device, args.batch_size, args.max_windows)
        mse_by_horizon[h] = (mse, n)
        print(f"  horizon={h}: MSE={format_float(mse)} windows={n}", flush=True)

    latents = collect_latents(model, val_set, device, args.batch_size, args.max_windows)
    print(f"  collected latents shape={tuple(latents.shape)}", flush=True)

    report = build_report(
        checkpoint=args.checkpoint,
        config_path=args.config,
        data_root=src_label,
        device=device,
        mse_by_horizon=mse_by_horizon,
        latents=latents,
        latent_dim=cfg.latent_dim,
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"\nrapport ecrit dans : {args.report}")


if __name__ == "__main__":
    main()
