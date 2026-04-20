#!/usr/bin/env python3
"""Point d'entrée — entraîne le WorldModel JEPA V1."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.world_model.config.config import WorldModelConfig
from src.world_model.models.world_model import WorldModel
from src.world_model.data.synthetic import make_dataloader
from src.world_model.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="World Model JEPA V1 — entraînement")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--samples", type=int, default=10000, help="Nombre de transitions synthétiques")
    parser.add_argument("--log-dir", type=str, default="runs/world_model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  World Model JEPA V1")
    print("=" * 60)

    # Config
    cfg = WorldModelConfig.from_yaml(args.config)
    print(f"Config       : {args.config}")
    print(f"obs_shape    : {cfg.obs_shape}")
    print(f"latent_dim   : {cfg.latent_dim}")
    print(f"batch_size   : {cfg.batch_size}")
    print(f"max_epochs   : {cfg.max_epochs}")
    print(f"samples      : {args.samples}")
    print("=" * 60)

    # Données synthétiques
    print("Génération des données synthétiques...")
    loader = make_dataloader(cfg, n_samples=args.samples)
    print(f"  {args.samples} transitions, {len(loader)} batches/epoch")

    # Modèle
    model = WorldModel(cfg)

    # Entraînement
    trainer = Trainer(model, cfg, log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir)
    trainer.fit(loader)


if __name__ == "__main__":
    main()
