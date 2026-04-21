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

## Expérience — λ_var = 0.15 (vs 0.04 baseline)

Pour tester si la régularisation plus forte corrige le collapse latent identifié ci-dessus, second run identique avec `lambda_var: 0.15` (config `configs/carracing_hv.yaml`, seed 1).

| Métrique | Baseline `λ_var=0.04` | `λ_var=0.15` | Effet |
|---|---:|---:|:---:|
| 1-step MSE | 0.0087 | 0.0132 | ×1.5 (pire) |
| Rollout h=5 | 0.042 | 0.080 | ×1.9 (pire) |
| Rollout h=10 | 0.141 | **7.15** | ×50 (cassé) |
| Rollout h=20 | 119 | **2.7×10⁷** | divergence |
| Variance latente moy | 0.076 | 0.134 | +76% (mieux) |
| Effective rank | 14.7 / 256 | 16.2 / 256 | +10% (mieux) |

**Conclusion honnête — expérience utile mais négative** :
- La reg fait son boulot primaire : variance latente ↑ +76%, rank effectif ↑ légèrement.
- Mais le coût est lourd : la prédiction se dégrade à toutes les échelles, le rollout long horizon devient instable au point de diverger numériquement (variance latente trop "étalée" → erreurs auto-amplifiées en rollout).
- **Le collapse latent n'est probablement pas la racine du problème**. Avec une politique aléatoire sur CarRacing, le signal visuel manque de diversité utile : forcer le latent à s'étaler artificiellement n'apporte pas de structure prédictible.

**Vraie piste suivante** : remplacer la politique aléatoire par une heuristique steering ou un agent PPO pré-entraîné, plutôt que de tuner λ_var.

---

## Expérience V1.6 — politique heuristique (vs random)

Suite logique de l'expé précédente : tester si une politique structurée corrige le rollout long horizon. Implémenté `--policy heuristic` dans `train.py` : steering = `0.6·sin(0.05·t) + 0.2·noise`, gas ∈ [0.4, 0.6], pas de frein. Trajectoires lisses, oscillation lente.

| Métrique | Baseline (random) | V1.6 (heuristic) | Effet |
|---|---:|---:|:---:|
| 1-step MSE | 0.0087 | 0.038 | ×4 (pire) |
| Rollout h=5 | 0.042 | 0.164 | ×4 (pire) |
| Rollout h=10 | 0.141 | 0.198 | ×1.4 (pire) |
| Rollout h=20 | **119** | **0.170** | **×700 (mieux)** |
| Effective rank | 14.7 | 13.7 | -7% |

**Découverte importante — trade-off précision vs stabilité** :

La random policy produit du bruit haute-fréquence sur les actions → faible signal corrélé entre obs successives → 1-step facile à apprendre mais erreur auto-amplifiée en rollout (compounding).

La heuristique produit des trajectoires structurées (steering oscillant) → contenu visuel plus dynamique mais corrélé temporellement → 1-step plus dur mais **rollout long horizon stable** (pas de divergence numérique).

**Conclusion** : pour un world model utilisable en planning, la heuristique gagne — un MSE constant à 0.17 sur tout l'horizon est exploitable, contrairement à la divergence à 10⁷. Le 1-step seul n'est pas un bon proxy de l'utilité du modèle.

**Pistes V2 prioritaires** :
- ~~Mixer random + heuristique pour avoir les deux régimes dans le dataset~~ ✅ V1.7
- Training avec teacher-forced rollout 5-step (la loss elle-même propage les erreurs)
- Agent PPO pré-entraîné pour des trajectoires near-optimal

---

## Expérience V1.7 — politique mixte (50/50 random + heuristic)

Hypothèse : combiner les deux régimes pour cumuler la précision 1-step de random et la stabilité long-horizon de heuristic.

| Métrique | Random | Heuristic | **Mixed** |
|---|---:|---:|---:|
| 1-step MSE | **0.0087** | 0.038 | 0.030 |
| Rollout h=5 | **0.042** | 0.164 | 0.103 |
| Rollout h=10 | **0.141** | 0.198 | 0.160 |
| Rollout h=20 | 119 | **0.170** | 0.229 |
| Variance latente moy | 0.076 | 0.059 | **0.150** |
| Effective rank | 14.7 | 13.7 | **23.1** (+57%) |

**Verdict — la mixte est globalement la plus utile** :
- **Meilleur effective rank** : 23.1 / 256, soit +57% vs random et +69% vs heuristic. Le latent est sensiblement plus riche.
- Rollout long horizon stable (0.23 à h=20, pas de divergence).
- Coût : 1-step et h=5 dégradés vs random (×3-4), mais sans atteindre la dégradation pure de heuristic.

**Insight** : la diversité d'actions (random) génère des observations diverses qui forcent l'encodeur à utiliser plus de dimensions latentes ; la structure (heuristic) garantit la stabilité du rollout. Les deux signaux sont complémentaires, pas redondants.

**Recommandation** : pour un world model destiné au planning, V1.7 (mixed) > V1.5 (random) > V1.6 (heuristic) en utilité globale.

---

## V1.8 — teacher-forced rollout (k=5)

Solution architecturale au compounding error : la loss elle-même propage les erreurs sur 5 steps.

- Nouveau `WorldModel.forward_rollout(obs_seq, action_seq)` : à chaque step k, prédit z_{k+1} depuis le z_k **prédit** (pas l'encodage réel) et compare à `encode(obs_{k+1})` en stop-gradient. Loss = moyenne MSE sur K.
- Nouveau `SequenceWindowDataset` : expose des fenêtres de K+1 obs consécutives, filtrées des passages cross-`done`.
- Flag `--rollout-k 5` dans `train.py` (combiné avec `--policy mixed` pour la diversité).
- Coût : ~3× plus lent par epoch (12 s vs 3.5 s).

**Synthèse 4-way (eval seed différents, 2000 transitions chacun)** :

| Métrique | V1.5 random | V1.6 heuristic | V1.7 mixed | **V1.8 rollout-5** |
|---|---:|---:|---:|---:|
| 1-step MSE | **0.0087** | 0.038 | 0.030 | 0.013 |
| Rollout h=5 | 0.042 | 0.164 | 0.103 | **0.021** |
| Rollout h=10 | 0.141 | 0.198 | 0.160 | **0.028** |
| Rollout h=20 | 119 | 0.170 | 0.229 | **0.038** |
| Effective rank | 14.7 | 13.7 | **23.1** | 20.6 |

**Résultat clé — MSE quasi-plat de h=1 à h=20 (×3), vs ×14 000 pour la baseline random** :
- Le compounding error est éliminé.
- Le 1-step est correct (0.013, entre random 0.0087 et heuristique 0.038).
- L'effective rank reste élevé (20.6, +40% vs random).

**V1.8 est le checkpoint à utiliser pour le planning** : c'est le seul qui prédit de façon fiable jusqu'à h=20.

**Vraies pistes V3** :
- Rollout k variable (curriculum 1→10) au lieu de fixe à 5
- DDP pour réduire le surcoût du rollout multi-step
- Encoder ViT (le CNN sature)
- Planning CEM/MPC réel sur ce world model

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

## Pistes V2 (résolues durant la nuit du 21–22 avril 2026)

- ~~Dataset réel (Gymnasium / CarRacing / Atari) au lieu de synthétique~~ ✅ V1.5
- ~~Évaluation : rollouts multi-step, variance/rang effectif du latent~~ ✅ V1.5
- ~~Corriger le collapse du latent (λ_var plus fort, politique non-aléatoire)~~ → expé λ_var=0.15 NÉG, V1.7 mixed +57% rank
- ~~Training multi-step (teacher-forced rollout) pour stabiliser h>10~~ ✅ V1.8
- Planification CEM/MPC à partir de `predict_next()`
- Encoder ViT à la place du CNN
- DDP (au lieu de DP) pour vraiment scaler multi-GPU
- t-SNE / UMAP du latent sur trajectoires CarRacing

---

## Licence

Voir le repo principal pour les conditions d'usage.
