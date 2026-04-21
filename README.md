# World Model JEPA — V1 / V1.5

World model inspiré de [JEPA (Yann LeCun)](https://openreview.net/forum?id=BZ5a1r-kVsf) :
prédiction dans l'**espace latent**, pas reconstruction pixel.

Brique d'un projet robot long terme par [Patrice Huetz](https://github.com/phuetz)
(le repo [GitNexus](https://github.com/phuetz) en est une autre brique).

---

## Architecture

```
   obs_t ──► ObservationEncoder ──► z_t ──┐
                                          ├──► LatentDynamicsModel ──► z_pred
   action_t ──► ActionEncoder ──► a_enc ──┘                              │
                                                                         ▼
   obs_{t+1} ──► ObservationEncoder ──► z_target  (stop-gradient)     MSE + reg
```

| Composant | Rôle | Détail |
|---|---|---|
| `ObservationEncoder` | image → latent | CNN 4 couches, projection linéaire vers `latent_dim` |
| `ActionEncoder` | action → latent | MLP 2 couches |
| `LatentDynamicsModel` | (z, a) → z′ | MLP 3 couches |
| `IsotropicLatentRegularizer` | empêche le collapse du latent | inspiré VICReg (variance + covariance + mean) |
| Têtes optionnelles | reward, done | linéaires depuis z′ |

**Loss totale** = `MSE(z_pred, z_target)` + régularisation isotrope + (reward_loss) + (done_loss BCE)

---

## Quickstart

```bash
git clone https://github.com/phuetz/world-model.git
cd world-model
pip install -r requirements.txt

# CUDA si GPU NVIDIA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Entraînement (dataset synthétique généré à la volée)
python scripts/train.py --config configs/default.yaml --samples 100000

# Entraînement sur un vrai env Gymnasium (V1.5)
pip install "gymnasium[box2d]"
python scripts/train.py --env CarRacing-v3 --samples 50000

# Évaluation d'un checkpoint
python scripts/eval.py --checkpoint checkpoints_carracing/epoch_0100.pt --env CarRacing-v3 --samples 2000
```

Sortie : checkpoints dans `checkpoints/epoch_NNNN.pt`, logs TensorBoard dans `runs/world_model/`.

```bash
tensorboard --logdir runs/
# http://localhost:6006
```

---

## Configuration (`configs/default.yaml`)

```yaml
obs_shape: [3, 64, 64]    # observations image
action_dim: 4
latent_dim: 256
encoder_type: conv
hidden_dim: 512
learning_rate: 0.0003
batch_size: 1024          # 64 sur CPU, 256+ sur RTX 3090
max_epochs: 100
lambda_var: 0.04
lambda_cov: 0.04
lambda_mean: 0.01
use_reward_head: true
use_done_head: true
```

---

## Multi-GPU

Le `Trainer` détecte `torch.cuda.device_count()` et wrappe automatiquement le modèle
dans `nn.DataParallel` si plusieurs GPUs sont visibles. Aucune CLI à passer.

Pour un modèle aussi petit (~2.5M params), le surcoût scatter/gather de DP est
non-négligeable et n'apporte un vrai gain qu'avec des batchs très grands ou un
modèle sensiblement plus gros (encoder ViT, latent_dim > 1024…).

---

## Résultats (training synthétique)

| Setup | GPUs | samples | bs | epochs | s/epoch | `loss_pred` final |
|---|---|---|---|---|---|---|
| CPU (G7 PT) | – | 1 000 | 64 | 100 | – | ~0.02 |
| 1× RTX 3090 | 1 | 100 000 | 256 | 100 | 8 | **0.0058** |
| 2× RTX 3090 (DP) | 2 | 200 000 | 1024 | 100 | 14 | **0.0021** |

Le checkpoint `checkpoints/epoch_0100.pt` correspond au dernier run multi-GPU.

---

## Évaluation V1.5 — CarRacing-v3

V1.5 passe du synthétique à de vraies trajectoires Gymnasium (CarRacing-v3, politique aléatoire). 50 000 transitions collectées, 100 epochs en 3.5 s/epoch sur 2× RTX 3090.

**MSE latent par horizon** (eval sur 2 000 transitions, seed différent du train) :

| Horizon | MSE latent | Points valides |
|---:|---:|---:|
| 1  | **0.0087** | 2 000 |
| 5  | 0.0422 | 1 990 |
| 10 | 0.1412 | 1 980 |
| 20 | **119.68** | 1 960 |

**Stats latent** :
- Variance moyenne par dimension : 0.076 (min 0.008, max 0.277)
- **Effective rank : 14.7 / 256 (5.8%)**

**Analyse honnête** :
- Le 1-step eval (0.0087) colle au loss_pred du training (0.0084) → pas de leakage, métrique saine.
- Le rollout à h=20 **explose** (119) : compounding error classique, pas de mécanisme de robustesse (bruit, teacher forcing, clipping du latent).
- **Le latent collapse** : 14.7 dim utiles sur 256. Soit la régularisation VICReg (`λ_var=0.04`, `λ_cov=0.04`) est trop faible, soit la politique aléatoire sur CarRacing produit des obs trop similaires pour forcer un latent riche.
- La `loss_total` domine par `loss_reward` (0.1) : les obs random → rewards presque tous nuls, la tête reward apprend peu.

**Pistes concrètes** :
- Monter `λ_var` à 0.1–0.25 pour contrer le collapse
- Remplacer la politique aléatoire par un agent PPO pré-entraîné ou une heuristique steering
- Ajouter un horizon de training >1-step (teacher-forced rollout) pour améliorer le long terme

---

## Arborescence

```
world-model/
├── configs/default.yaml          # hyperparamètres
├── requirements.txt
├── scripts/
│   ├── train.py                  # point d'entrée entraînement
│   └── eval.py                   # évaluation multi-step + stats latent
├── checkpoints/                  # checkpoints (runs synthétiques)
├── checkpoints_carracing/        # checkpoints CarRacing V1.5
├── runs/                         # logs TensorBoard
├── CLAUDE.md                     # instructions pour Claude Code
└── src/world_model/
    ├── config/config.py          # WorldModelConfig (dataclass + from_yaml)
    ├── data/
    │   ├── synthetic.py          # dataset synthétique pour V1
    │   └── gym_env.py            # collecte Gymnasium pour V1.5
    ├── models/
    │   ├── encoder.py            # ObservationEncoder (CNN) + ActionEncoder
    │   ├── dynamics.py           # LatentDynamicsModel + têtes reward/done
    │   ├── regularizer.py        # IsotropicLatentRegularizer (VICReg)
    │   └── world_model.py        # WorldModel orchestrateur
    └── training/trainer.py       # boucle entraînement + TensorBoard + checkpoints
```

---

## Pistes V2

- ~~Dataset réel (Gymnasium / CarRacing / Atari) au lieu de synthétique~~ ✅ V1.5
- ~~Évaluation : rollouts multi-step, variance/rang effectif du latent~~ ✅ V1.5
- Corriger le collapse du latent (λ_var plus fort, politique non-aléatoire)
- Training multi-step (teacher-forced rollout) pour stabiliser h>10
- Planification CEM/MPC à partir de `predict_next()`
- Encoder ViT à la place du CNN
- DDP (au lieu de DP) pour vraiment scaler multi-GPU
- t-SNE / UMAP du latent sur trajectoires CarRacing

---

## Licence

Voir le repo principal pour les conditions d'usage.
