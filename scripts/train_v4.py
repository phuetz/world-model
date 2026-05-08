"""V4 trainer — réutilise la pipeline V3 (Conv/Transformer + DDPTrainer)
mais collecte le dataset depuis un env Gymnasium au lieu d'un dataset vidéo.

Usage smoke (1 GPU) :
  python scripts/train_v4.py --config configs/v4_lunarlander.yaml \\
      --env LunarLanderContinuous-v3 --n-episodes 50 --max-episode-len 200 \\
      --gpus 1 --max-steps 50

Usage full (1 GPU) :
  python scripts/train_v4.py --config configs/v4_lunarlander.yaml \\
      --env LunarLanderContinuous-v3 --n-episodes 500 --max-episode-len 300 \\
      --policy heuristic --gpus 1
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Workaround Win11 (cf. train_v3.py)
os.environ.setdefault("USE_LIBUV", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from world_model.config.config import WorldModelConfig  # noqa: E402
from world_model.data.gym_video_dataset import (  # noqa: E402
    GymVideoDataset, make_gym_video_dataloader,
)
from world_model.models.world_model import WorldModel  # noqa: E402
from world_model.training.ddp_trainer import DDPTrainer  # noqa: E402


def _worker(rank: int, world_size: int, args) -> None:
    os.environ["USE_LIBUV"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if world_size > 1:
        from world_model.training.ddp_trainer import setup_ddp
        setup_ddp(args.backend)

    cfg = WorldModelConfig.from_yaml(args.config)

    # Collecte une seule fois le dataset (env steps coûteux), partage train/val
    if rank == 0:
        print(f"[train_v4] collecting {args.n_episodes} episodes from {args.env} "
              f"(policy={args.policy}, max_len={args.max_episode_len})...", flush=True)

    full_ds = GymVideoDataset(
        cfg, env_id=args.env, n_episodes=args.n_episodes,
        max_episode_len=args.max_episode_len, policy=args.policy,
        seed=args.seed, stride=args.stride,
    )

    if rank == 0:
        print(f"[train_v4] collected {full_ds.n_clips} episodes / {len(full_ds)} windows",
              flush=True)

    train_loader = make_gym_video_dataloader(
        cfg, env_id=args.env, n_episodes=args.n_episodes,
        max_episode_len=args.max_episode_len, policy=args.policy,
        seed=args.seed, stride=args.stride,
        split="train", val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        distributed_sampler=(world_size > 1),
        shared_dataset=full_ds,
    )
    val_loader = make_gym_video_dataloader(
        cfg, env_id=args.env, n_episodes=args.n_episodes,
        max_episode_len=args.max_episode_len, policy=args.policy,
        seed=args.seed, stride=args.stride,
        split="val", val_ratio=args.val_ratio,
        num_workers=max(0, args.num_workers // 2),
        distributed_sampler=(world_size > 1),
        shared_dataset=full_ds,
    )

    if rank == 0:
        print(f"[train_v4] train_windows={len(train_loader.dataset)} "
              f"val_windows={len(val_loader.dataset)}", flush=True)

    model = WorldModel(cfg)
    trainer = DDPTrainer(
        model, cfg, train_loader, val_loader=val_loader,
        log_dir=args.log_dir, checkpoint_dir=args.ckpt_dir,
        backend=args.backend,
    )
    try:
        trainer.fit(max_steps=args.max_steps)
    finally:
        trainer.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--env", default="LunarLanderContinuous-v3")
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--max-episode-len", type=int, default=200)
    p.add_argument("--policy", default="random", choices=["random", "heuristic"])
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/v4_gym")
    p.add_argument("--ckpt-dir", default="checkpoints_v4_gym")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0,
                   help="V4 collecte tout en RAM, 0 worker recommandé")
    p.add_argument("--backend", type=str, default="gloo")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--master-port", type=int, default=29501)
    args = p.parse_args()

    world_size = max(1, args.gpus)
    if world_size == 1:
        _worker(0, 1, args)
    else:
        mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
