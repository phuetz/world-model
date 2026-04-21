# World Model JEPA — V1

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

## Arborescence

```
world-model/
├── configs/default.yaml          # hyperparamètres
├── requirements.txt
├── scripts/train.py              # point d'entrée
├── checkpoints/                  # checkpoints sauvegardés (toutes les 10 epochs)
├── runs/                         # logs TensorBoard
├── CLAUDE.md                     # instructions pour Claude Code
└── src/world_model/
    ├── config/config.py          # WorldModelConfig (dataclass + from_yaml)
    ├── data/synthetic.py         # dataset synthétique pour V1
    ├── models/
    │   ├── encoder.py            # ObservationEncoder (CNN) + ActionEncoder
    │   ├── dynamics.py           # LatentDynamicsModel + têtes reward/done
    │   ├── regularizer.py        # IsotropicLatentRegularizer (VICReg)
    │   └── world_model.py        # WorldModel orchestrateur
    └── training/trainer.py       # boucle entraînement + TensorBoard + checkpoints
```

---

## Pistes V2

- Dataset réel (Gymnasium / CarRacing / Atari) au lieu de synthétique
- Évaluation : rollouts multi-step, variance/rang effectif du latent, t-SNE
- Planification CEM/MPC à partir de `predict_next()`
- Encoder ViT à la place du CNN
- DDP (au lieu de DP) pour vraiment scaler multi-GPU

---

## Licence

Voir le repo principal pour les conditions d'usage.
