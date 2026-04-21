#!/usr/bin/env python3
"""Evaluation du WorldModel JEPA sur des transitions Gymnasium."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.world_model.config.config import WorldModelConfig
from src.world_model.data.gym_env import GymTransitionDataset
from src.world_model.models.world_model import WorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation du WorldModel JEPA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_carracing/epoch_0100.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--env", type=str, default="CarRacing-v3")
    parser.add_argument("--samples", type=int, default=2000, help="Nombre de transitions d'evaluation")
    parser.add_argument("--horizons", type=str, default="1,5,10,20", help="Liste CSV d'horizons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=str, default="eval_report.md")
    return parser.parse_args()


def parse_horizons(value: str) -> List[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("--horizons doit contenir des entiers positifs, ex: 1,5,10,20")
    return horizons


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_model(cfg: WorldModelConfig, checkpoint_path: str, device: torch.device) -> WorldModel:
    model = WorldModel(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint invalide: impossible de trouver un state_dict dans {checkpoint_path}")

    model.load_state_dict(strip_module_prefix(state))
    model.eval()
    return model


def encode_batches(model: WorldModel, obs: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    latents: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(obs), batch_size):
            batch = obs[start : start + batch_size].to(device)
            latents.append(model.encode(batch).detach().cpu())
    return torch.cat(latents, dim=0)


def one_step_mse(model: WorldModel, dataset: GymTransitionDataset, device: torch.device, batch_size: int) -> float:
    losses: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            obs_t = dataset.obs_t[start : start + batch_size].to(device)
            actions = dataset.actions[start : start + batch_size].to(device)
            obs_tp1 = dataset.obs_tp1[start : start + batch_size].to(device)

            z_t = model.encode(obs_t)
            z_target = model.encode(obs_tp1)
            z_pred = model.predict_next(z_t, actions)
            mse_per_sample = F.mse_loss(z_pred, z_target, reduction="none").mean(dim=1)
            losses.append(mse_per_sample.detach().cpu())
    return torch.cat(losses).mean().item()


def valid_start_indices(dones: torch.Tensor, horizon: int) -> torch.Tensor:
    max_start = len(dones) - horizon + 1
    if max_start <= 0:
        return torch.empty(0, dtype=torch.long)

    valid: List[int] = []
    for start in range(max_start):
        if not bool(dones[start : start + horizon].any()):
            valid.append(start)
    return torch.tensor(valid, dtype=torch.long)


def rollout_mse(
    model: WorldModel,
    dataset: GymTransitionDataset,
    horizon: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, int]:
    starts = valid_start_indices(dataset.dones, horizon)
    if starts.numel() == 0:
        return float("nan"), 0

    losses: List[torch.Tensor] = []
    with torch.no_grad():
        for offset in range(0, starts.numel(), batch_size):
            idx = starts[offset : offset + batch_size]
            z = model.encode(dataset.obs_t[idx].to(device))

            for step in range(horizon):
                action_idx = idx + step
                actions = dataset.actions[action_idx].to(device)
                z = model.predict_next(z, actions)

            target_idx = idx + horizon - 1
            z_target = model.encode(dataset.obs_tp1[target_idx].to(device))
            mse_per_sample = F.mse_loss(z, z_target, reduction="none").mean(dim=1)
            losses.append(mse_per_sample.detach().cpu())

    return torch.cat(losses).mean().item(), int(starts.numel())


def effective_rank(latents: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(latents.float())
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probs = singular_values / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return torch.exp(entropy).item()


def format_float(value: float) -> str:
    if value != value:
        return "n/a"
    return f"{value:.6f}"


def build_report(
    checkpoint: str,
    env_id: str,
    samples: int,
    device: torch.device,
    mse_by_horizon: Dict[int, tuple[float, int]],
    latents: torch.Tensor,
    latent_dim: int,
) -> str:
    variances = latents.float().var(dim=0, unbiased=False)
    var_mean = variances.mean().item()
    var_min = variances.min().item()
    var_max = variances.max().item()
    rank_eff = effective_rank(latents)
    rank_pct = 100.0 * rank_eff / latent_dim if latent_dim > 0 else 0.0

    rows = ["| Horizon | MSE latent | Points valides |", "|---:|---:|---:|"]
    for horizon, (mse, count) in mse_by_horizon.items():
        rows.append(f"| {horizon} | {format_float(mse)} | {count} |")

    return "\n".join(
        [
            "# Rapport d'evaluation World Model JEPA",
            "",
            f"- Checkpoint: `{checkpoint}`",
            f"- Env: `{env_id}`",
            f"- Samples: `{samples}`",
            f"- Device: `{device}`",
            f"- Date: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## MSE par horizon",
            "",
            *rows,
            "",
            "## Statistiques du latent",
            "",
            f"- Variance moyenne par dimension: `{var_mean:.6f}`",
            f"- Variance min par dimension: `{var_min:.6f}`",
            f"- Variance max par dimension: `{var_max:.6f}`",
            f"- Effective rank: `{rank_eff:.2f}` / `{latent_dim}` (`{rank_pct:.2f}%`)",
            "",
            "## Interpretation",
            "",
            f"Rank {rank_eff:.1f}/{latent_dim} -> le modele utilise {rank_pct:.1f}% de sa capacite latente.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = WorldModelConfig.from_yaml(args.config)
    model = load_model(cfg, args.checkpoint, device)

    dataset = GymTransitionDataset(cfg, args.env, n_samples=args.samples, seed=args.seed)
    batch_size = max(1, int(cfg.batch_size))

    mse_by_horizon: Dict[int, tuple[float, int]] = {}
    for horizon in horizons:
        if horizon == 1:
            mse_by_horizon[horizon] = (one_step_mse(model, dataset, device, batch_size), len(dataset))
        else:
            mse_by_horizon[horizon] = rollout_mse(model, dataset, horizon, device, batch_size)

    latents = encode_batches(model, dataset.obs_t, device, batch_size)
    report = build_report(
        checkpoint=args.checkpoint,
        env_id=args.env,
        samples=args.samples,
        device=device,
        mse_by_horizon=mse_by_horizon,
        latents=latents,
        latent_dim=cfg.latent_dim,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
