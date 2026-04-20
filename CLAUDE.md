# World Model JEPA V1 — Instructions pour Claude Code

## Contexte du projet

World Model inspiré de JEPA (Yann LeCun) pour un projet robot long terme.
Principe : prédiction dans l'espace latent, pas reconstruction pixel.

Développé par Patrice Huetz avec Claude Code (PC principal : Minisforum G7 PT).
Ce PC (2× RTX 3090) est dédié à l'entraînement GPU.

## Ce que tu dois faire sur ce PC

### 1. Vérifier CUDA
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Si CUDA n'est pas détecté, installe PyTorch avec CUDA :
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'entraînement
```bash
python scripts/train.py --config configs/default.yaml --samples 100000
```

Avec 2× RTX 3090, chaque epoch devrait prendre < 5 secondes.

### 4. Visualiser les courbes (optionnel)
```bash
tensorboard --logdir runs/
```
Ouvre http://localhost:6006 dans le navigateur.

### 5. Quand l'entraînement est terminé
- Les checkpoints sont dans `checkpoints/`
- Committe et push le meilleur checkpoint :
```bash
git add checkpoints/epoch_0100.pt
git commit -m "feat: checkpoint epoch 100 — 3090 training"
git push
```

## Architecture du projet

```
world-model/
├── configs/default.yaml          ← hyperparamètres (modifie ici)
├── requirements.txt
└── src/world_model/
    ├── config/config.py          ← WorldModelConfig (dataclass + from_yaml)
    └── models/
        ├── encoder.py            ← ObservationEncoder (CNN) + ActionEncoder
        ├── dynamics.py           ← LatentDynamicsModel + têtes reward/done
        ├── regularizer.py        ← IsotropicLatentRegularizer (VICReg)
        └── world_model.py        ← WorldModel orchestrateur (forward_step)
    ├── data/
    │   └── synthetic.py          ← Dataset synthétique pour V1
    └── training/
        └── trainer.py            ← Boucle entraînement + TensorBoard + checkpoints
```

## Hyperparamètres (configs/default.yaml)

```yaml
obs_shape: [3, 64, 64]     # taille des observations image
action_dim: 4               # dimension des actions
latent_dim: 256             # dimension de l'espace latent
encoder_type: conv          # conv | vit
hidden_dim: 512
learning_rate: 0.0003
batch_size: 64              # augmente à 256 sur les 3090
max_epochs: 100
```

Pour profiter des 3090, augmente le batch_size :
```bash
# Edite configs/default.yaml : batch_size: 256
python scripts/train.py --config configs/default.yaml --samples 100000
```

## Résultats attendus

Sur CPU (G7 PT) avec 1000 samples :
- loss_pred epoch 1 : ~1.8
- loss_pred epoch 100 : ~0.02
- Convergence validée ✓

Sur GPU 3090 avec 100 000 samples : objectif loss_pred < 0.01

## Contact / repo principal

- Repo : https://github.com/phuetz/world-model
- Projet robot long terme de Patrice — GitNexus est une autre brique du même projet
- PC principal pour le dev : Minisforum G7 PT Windows (Claude Code actif là-bas)
