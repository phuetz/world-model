#!/usr/bin/env python3
"""Plan une politique CEM/MPC sur CarRacing avec un World Model entraîné.

Compare 3 politiques sur N épisodes :
- random  : env.action_space.sample()
- heuristic : politique scriptée (cf. gym_env._heuristic_carracing)
- cem     : CEM/MPC en latent (re-plan à chaque step, prend la 1ère action)
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.world_model.config.config import WorldModelConfig
from src.world_model.data.gym_env import _preprocess_obs, _heuristic_carracing
from src.world_model.models.world_model import WorldModel
from src.world_model.planning.cem import CEMPlanner, CEMConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Planning CEM/MPC sur un World Model entraîné")
    p.add_argument("--checkpoint", default="checkpoints_carracing_rollout5/epoch_0100.pt")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--env", default="CarRacing-v3")
    p.add_argument("--episodes", type=int, default=3, help="N épisodes par politique")
    p.add_argument("--max-steps", type=int, default=400, help="Cap par épisode")
    p.add_argument("--horizon", type=int, default=12, help="CEM horizon")
    p.add_argument("--n-samples", type=int, default=512, help="CEM samples par itération")
    p.add_argument("--n-elites", type=int, default=64)
    p.add_argument("--n-iterations", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", default="plan_report.md")
    p.add_argument("--policies", default="random,heuristic,cem",
                   help="Liste CSV des politiques à comparer")
    return p.parse_args()


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def load_model(cfg: WorldModelConfig, path: str, device: torch.device) -> WorldModel:
    model = WorldModel(cfg).to(device)
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state") if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(strip_module_prefix(state))
    model.eval()
    return model


def run_episode(env, policy_name: str, planner: CEMPlanner | None,
                model: WorldModel | None, cfg: WorldModelConfig, device: torch.device,
                rng: np.random.Generator, max_steps: int, seed: int) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    if planner is not None:
        planner.last_mean = None  # reset MPC warm-start

    for step in range(max_steps):
        if policy_name == "random":
            action = env.action_space.sample()
        elif policy_name == "heuristic":
            action = _heuristic_carracing(step, env, rng)
        elif policy_name == "cem":
            C, H_, W_ = cfg.obs_shape
            obs_t = _preprocess_obs(np.asarray(obs), H_, W_).unsqueeze(0).to(device)
            z = model.encode(obs_t).squeeze(0)
            plan = planner.plan(z)
            action = plan[0].detach().cpu().numpy().astype(np.float32)
            # action de l'env: shape 3 pour CarRacing (action_dim 4 → tronque)
            action = action[: env.action_space.shape[0]]
            action = np.clip(action, env.action_space.low, env.action_space.high)
        else:
            raise ValueError(policy_name)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return {"return": total_reward, "length": steps}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = WorldModelConfig.from_yaml(args.config)
    model = load_model(cfg, args.checkpoint, device)

    cem_cfg = CEMConfig(
        horizon=args.horizon, n_samples=args.n_samples,
        n_elites=args.n_elites, n_iterations=args.n_iterations,
    )
    planner = CEMPlanner(model, action_dim=cfg.action_dim, cfg=cem_cfg, device=device)

    import gymnasium as gym
    env = gym.make(args.env)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    rng = np.random.default_rng(args.seed)
    env.action_space.seed(args.seed)

    results: Dict[str, List[Dict[str, float]]] = {p: [] for p in policies}
    for episode in range(args.episodes):
        for pol in policies:
            seed = args.seed + episode * 1000
            out = run_episode(env, pol, planner if pol == "cem" else None,
                              model, cfg, device, rng, args.max_steps, seed)
            results[pol].append(out)
            print(f"  ep{episode}  {pol:9s} -> return={out['return']:8.2f}  length={out['length']}")

    env.close()

    # Rapport
    lines = [
        "# Rapport de planning World Model",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Env: `{args.env}`",
        f"- Episodes: `{args.episodes}` × `{args.max_steps}` steps max",
        f"- CEM: horizon={args.horizon}, samples={args.n_samples}, elites={args.n_elites}, iters={args.n_iterations}",
        f"- Date: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Résultats par politique",
        "",
        "| Politique | Return moyen | Return médian | Length moyenne |",
        "|---|---:|---:|---:|",
    ]
    for pol in policies:
        rs = [r["return"] for r in results[pol]]
        ls = [r["length"] for r in results[pol]]
        rs_t = torch.tensor(rs)
        lines.append(
            f"| {pol} | {rs_t.mean().item():.2f} | {rs_t.median().item():.2f} | {sum(ls)/len(ls):.0f} |"
        )

    report = "\n".join(lines) + "\n"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    main()
