#!/usr/bin/env python3
"""CEM open-loop inverse planning sur world-model V3.

Pas d'environnement closed-loop pour le dataset video. À la place : pour chaque paire
(z_0, z_T_target) tirée du val set, CEM cherche la séquence d'actions [a_0..a_{T-1}]
qui amène le rollout(z_0, actions)[T] le plus proche de z_T_target.

Métrique : ratio MSE final (après CEM) / MSE initial (z_0 vs z_T_target).
- ratio < 0.5 = dynamics inversible utile (CEM trouve une trajectoire qui converge)
- ratio ~ 1.0 = dynamics inutilisable pour planning
- ratio > 1.0 = bug

Usage :
  python scripts/plan_v3.py --checkpoint checkpoints_v3_video/epoch_0050.pt
"""
from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from world_model.config.config import WorldModelConfig  # noqa: E402
from world_model.data.video_dataset import VideoClipDataset, split_clips  # noqa: E402
from world_model.models.world_model import WorldModel  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from scripts.eval import strip_module_prefix, format_float  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/v3_video.yaml")
    p.add_argument("--data", default="data/v3_video")
    p.add_argument("--n-pairs", type=int, default=100)
    p.add_argument("--horizon", type=int, default=8, help="T (au plus seq_len) — horizon du rollout cible")
    p.add_argument("--cem-iters", type=int, default=4)
    p.add_argument("--cem-samples", type=int, default=512)
    p.add_argument("--cem-elite", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default="plan_report_v3.md")
    return p.parse_args()


def load_model(cfg: WorldModelConfig, ckpt_path: str, device: torch.device) -> WorldModel:
    model = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        state = ckpt
    model.load_state_dict(strip_module_prefix(state))
    model.eval()
    return model


@torch.no_grad()
def cem_actions(
    model: WorldModel,
    z0: torch.Tensor,
    z_target: torch.Tensor,
    horizon: int,
    n_samples: int,
    n_iters: int,
    n_elite: int,
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """CEM minimisant MSE(rollout(z0, actions)[T], z_target).

    Distribution gaussienne sur (T, action_dim), itérations refit sur les `n_elite` meilleurs.
    z0, z_target : (latent_dim,) — version mono-batch.
    Retourne (best_actions [T, action_dim], best_mse, init_mse).
    """
    init_mse = float(F.mse_loss(z0, z_target).item())

    # Stats initiales (uniforme dans range observed = std 1.0 si z normé bf16/fp32)
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    z0_b = z0.unsqueeze(0).expand(n_samples, -1)
    z_target_b = z_target.unsqueeze(0).expand(n_samples, -1)

    best_actions = torch.zeros(horizon, action_dim, device=device)
    best_mse = float("inf")
    for _ in range(n_iters):
        # Sample
        eps = torch.randn(n_samples, horizon, action_dim, device=device)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        # Rollout
        z = z0_b
        for t in range(horizon):
            z = model.predict_next(z, actions[:, t])
        # Score
        mse = F.mse_loss(z, z_target_b, reduction="none").mean(dim=1)
        # Elite
        elite_idx = mse.topk(n_elite, largest=False).indices
        elite_actions = actions[elite_idx]
        # Refit
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp_min(1e-3)
        # Track best
        cur_best = mse[elite_idx[0]].item()
        if cur_best < best_mse:
            best_mse = cur_best
            best_actions = actions[elite_idx[0]].clone()

    return best_actions, best_mse, init_mse


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = WorldModelConfig.from_yaml(args.config)
    model = load_model(cfg, args.checkpoint, device)

    _, val_ids = split_clips(args.data, val_ratio=0.05, seed=args.seed)
    val_set = VideoClipDataset(cfg, args.data, seq_len=cfg.seq_len, stride=2, clip_ids=val_ids)

    if args.horizon > cfg.seq_len:
        raise ValueError(f"horizon {args.horizon} > seq_len {cfg.seq_len}")

    H = args.horizon
    n_pairs = min(args.n_pairs, len(val_set))
    print(f"[plan_v3] val_windows={len(val_set)} n_pairs={n_pairs} H={H}", flush=True)

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    indices = torch.randperm(len(val_set), generator=rng)[:n_pairs].tolist()

    init_mses, final_mses, ratios = [], [], []
    with torch.no_grad():
        for k, i in enumerate(indices):
            obs_seq, action_seq, _, _ = val_set[i]
            obs_seq = obs_seq.to(device)
            z0 = model.encode(obs_seq[:1])[0]
            zT = model.encode(obs_seq[H : H + 1])[0]
            _, final_mse, init_mse = cem_actions(
                model, z0, zT, horizon=H,
                n_samples=args.cem_samples, n_iters=args.cem_iters,
                n_elite=args.cem_elite, action_dim=cfg.action_dim, device=device,
            )
            init_mses.append(init_mse)
            final_mses.append(final_mse)
            r = final_mse / init_mse if init_mse > 0 else float("nan")
            ratios.append(r)
            if k % 10 == 0:
                print(f"  [{k+1}/{n_pairs}] init={init_mse:.4f} final={final_mse:.4f} ratio={r:.3f}", flush=True)

    import statistics as st
    mean_init = sum(init_mses) / max(1, len(init_mses))
    mean_final = sum(final_mses) / max(1, len(final_mses))
    mean_ratio = sum(ratios) / max(1, len(ratios))
    median_ratio = st.median(ratios) if ratios else float("nan")

    report = "\n".join([
        "# Rapport CEM open-loop V3 (inverse planning)",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Config: `{args.config}`",
        f"- Dataset: `{args.data}`",
        f"- Device: `{device}`",
        f"- Date: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Paramètres",
        "",
        f"- Pairs: `{n_pairs}`  (windows tirées au hasard du val set)",
        f"- Horizon: `{H}`",
        f"- CEM samples / iters / elite: `{args.cem_samples}` / `{args.cem_iters}` / `{args.cem_elite}`",
        "",
        "## Résultats",
        "",
        f"- MSE initial moyen (z_0 vs z_T_target avant CEM): `{format_float(mean_init)}`",
        f"- MSE final moyen (après CEM): `{format_float(mean_final)}`",
        f"- **Ratio MSE_final / MSE_initial** moyen: `{format_float(mean_ratio)}`",
        f"- Ratio médian: `{format_float(median_ratio)}`",
        "",
        "## Interprétation",
        "",
        (
            "ratio < 0.5 = dynamics inversible utile pour planning ; "
            "ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation."
        ),
        "",
    ])
    Path(args.report).write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
