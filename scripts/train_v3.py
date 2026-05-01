"""Point d'entrée DDP pour le training V3 sur dataset vidéo Wan/SVD.

Lancement (pas de torchrun — libuv issue Win11) :
  python scripts/train_v3.py --config configs/v3_video.yaml --data data/v3_video --gpus 2

Smoke run :
  python scripts/train_v3.py --config configs/v3_video.yaml --data data/v3_video_smoke \\
      --gpus 2 --max-steps 20 --num-workers 0
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Workaround Win11 : PyTorch wheels Windows ne sont pas buildés avec libuv,
# le rendezvous tente de l'utiliser par défaut → DistStoreError.
os.environ.setdefault("USE_LIBUV", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from world_model.config.config import WorldModelConfig  # noqa: E402
from world_model.data.video_dataset import make_video_dataloader  # noqa: E402
from world_model.models.world_model import WorldModel  # noqa: E402
from world_model.training.ddp_trainer import DDPTrainer  # noqa: E402


def _worker(rank: int, world_size: int, args) -> None:
    os.environ["USE_LIBUV"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Init DDP avant tout DataLoader distribué
    if world_size > 1:
        from world_model.training.ddp_trainer import setup_ddp
        setup_ddp(args.backend)

    cfg = WorldModelConfig.from_yaml(args.config)

    train_loader = make_video_dataloader(
        cfg, root=args.data, split="train",
        num_workers=args.num_workers,
        distributed_sampler=(world_size > 1),
        blacklist_path=args.blacklist,
    )
    val_loader = make_video_dataloader(
        cfg, root=args.data, split="val",
        num_workers=max(0, args.num_workers // 2),
        distributed_sampler=(world_size > 1),
        blacklist_path=args.blacklist,
    )

    if rank == 0:
        ds = train_loader.dataset
        print(
            f"[train_v3] train_clips={ds.n_clips} train_windows={len(ds)} | val_windows={len(val_loader.dataset)}",
            flush=True,
        )

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
    p.add_argument("--data", required=True, type=str, help="root du dataset video")
    p.add_argument("--log-dir", default="runs/v3_video")
    p.add_argument("--ckpt-dir", default="checkpoints_v3_video")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--blacklist", type=str, default=None)
    p.add_argument("--backend", type=str, default="gloo")
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--master-port", type=int, default=29500)
    args = p.parse_args()

    world_size = max(1, args.gpus)
    if world_size == 1:
        _worker(0, 1, args)
    else:
        mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
