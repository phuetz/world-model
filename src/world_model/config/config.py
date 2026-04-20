"""WorldModelConfig — centralise tous les hyperparamètres."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import yaml


@dataclass
class WorldModelConfig:
    # Observation & action
    obs_shape: Tuple[int, int, int] = (3, 64, 64)
    action_dim: int = 4
    latent_dim: int = 256

    # Architecture
    encoder_type: str = "conv"   # "conv" | "vit"
    hidden_dim: int = 512

    # Entraînement
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 100

    # Régularisation isotrope
    lambda_var: float = 0.04
    lambda_cov: float = 0.04
    lambda_mean: float = 0.01

    # Têtes auxiliaires
    use_reward_head: bool = True
    use_done_head: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "WorldModelConfig":
        """Charge la config depuis un fichier YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "obs_shape" in data:
            data["obs_shape"] = tuple(data["obs_shape"])
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Sauvegarde la config dans un fichier YAML."""
        import dataclasses
        d = dataclasses.asdict(self)
        d["obs_shape"] = list(d["obs_shape"])
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False)
