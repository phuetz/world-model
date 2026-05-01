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
├── configs/
│   ├── default.yaml             ← V1/V2 (CarRacing 64×64, MLP dynamique)
│   ├── carracing_hv.yaml        ← V1.6 lambda_var élevé
│   └── v3_video.yaml            ← V3 (256×256, Transformer dynamique, video)
├── requirements.txt
├── scripts/
│   ├── train.py                 ← V1/V2 (CarRacing) — DataParallel
│   ├── train_v3.py              ← V3 (video) — mp.spawn launcher, USE_LIBUV=0, single/multi-GPU
│   ├── eval.py                  ← V1/V2 eval (Gym)
│   ├── eval_v3.py               ← V3 eval (video horizons + rank + compounding)
│   ├── plan.py                  ← V2 CEM/MPC planner
│   ├── plan_v3.py               ← V3 CEM open-loop inverse planning
│   └── dataset_v3/              ← V3 production pipeline (SVD-XT i2v)
│       ├── prompt_templates.py  ← 4 classes × pools, dedup hash
│       ├── stock_image.py       ← 4 generators procéduraux par classe
│       ├── optical_flow.py      ← Farneback action proxy 4D
│       ├── comfy_client.py      ← API ComfyUI (POST /prompt + poll /history)
│       ├── produce_dataset.py   ← Producer multi-server, work-queue async, watchdog
│       ├── qa_dataset.py        ← Contact sheet + stats + blacklist auto
│       └── workflows/
│           ├── sdxl_image.json  ← (legacy, pas utilisé en V3 SVD-XT-only)
│           └── svd_i2v.json     ← SVD-XT image→video API workflow
└── src/world_model/
    ├── config/config.py          ← WorldModelConfig (dataclass + from_yaml)
    │                                + V3 fields : dynamics_type, seq_len,
    │                                rollout_warmup_epochs, use_amp
    ├── models/
    │   ├── encoder.py            ← ObservationEncoder (Conv4 64×64) +
    │   │                           ObservationEncoderConv5 (V3 256×256) +
    │   │                           ActionEncoder + factory
    │   ├── dynamics.py           ← LatentDynamicsModel (MLP V1/V2) +
    │   │                           LatentDynamicsTransformer (V3 causal pre-norm) +
    │   │                           factory
    │   ├── regularizer.py        ← IsotropicLatentRegularizer (VICReg)
    │   └── world_model.py        ← WorldModel orchestrateur (forward_step / forward_rollout)
    ├── data/
    │   ├── synthetic.py          ← Dataset synthétique V1
    │   ├── gym_env.py            ← V2 CarRacing (Gymnasium)
    │   └── video_dataset.py      ← V3 VideoClipDataset lazy-load PIL + split clips
    ├── planning/cem.py           ← CEM/MPC batched GPU (V2 + V3)
    └── training/
        ├── trainer.py            ← V1/V2 trainer (DataParallel)
        └── ddp_trainer.py        ← V3 trainer (DDP gloo + bf16 + AdamW cosine warmup)
```

## Versions

| Version | Dataset | Encoder | Dynamics | latent_dim | Params | Status |
|---|---|---|---|---|---|---|
| V1.0-1.4 | Synthétique | Conv4 64×64 | MLP | 256 | 2.5M | archived |
| V1.5-1.7 | CarRacing-v3 random/heur/mixed | Conv4 64×64 | MLP | 256 | 2.5M | ✓ |
| V1.8 | CarRacing teacher-forced k=5 | Conv4 64×64 | MLP | 256 | 2.5M | ✓ — compounding éliminé |
| V2.0 | CarRacing + CEM/MPC planner | Conv4 64×64 | MLP | 256 | 2.5M | ✓ — CEM bat random |
| **V3** | **SVD-XT video stock 256×256 T=16** | **Conv5** | **Transformer 4×8×512** | **512** | **23.8M** | **en cours 2026-05-01** |

Cibles V3 vs V1.8 :
- MSE h=1 < 0.020 (V1.8 = 0.0135) ; partial 51 clips déjà à **0.007** ✓
- Compounding ratio MSE(h=16)/MSE(h=1) < ×2.0 (V1.8 = ×2.8 sur h=20) ; partial = **1.54** ✓
- **Effective rank > 80/512 (15%)** ← métrique principale (V1.8 = 8%)

## Pièges Win11 (DARKSTAR)

- **DDP 2-GPU plante** en ACCESS_VIOLATION 0xC0000005 sur Win11. Stratégie :
  single-GPU training, l'autre GPU pour ComfyUI inférence.
- **`torchrun` exige libuv** absent sur les wheels PyTorch Windows. Bypass via
  `mp.spawn` avec `os.environ.setdefault("USE_LIBUV", "0")` AVANT `import torch`.
  Voir `scripts/train_v3.py`.
- **Tailscale SSH server** pas supporté Windows → utiliser **OpenSSH Server natif**.
- `psutil.disk_usage(rel_path)` plante → utiliser `Path.resolve().anchor or os.getcwd()`.

## Lancer training V3

```bash
# Generation du dataset (parallel 2× ComfyUI servers, GPU 0 + GPU 1)
# (ComfyUI server #1 sur :8188 GPU 1, server #2 sur :8189 GPU 0)
python scripts/dataset_v3/prompt_templates.py --target 1500 --out scripts/dataset_v3/prompts.jsonl
python scripts/dataset_v3/produce_dataset.py \
  --prompts scripts/dataset_v3/prompts.jsonl \
  --servers 127.0.0.1:8188,127.0.0.1:8189 \
  --out data/v3_video --target 1500 --clip-length 25

# QA + blacklist
python scripts/dataset_v3/qa_dataset.py --root data/v3_video

# Training V3 (single-GPU, ~25 min sur 1500 clips)
python scripts/train_v3.py --config configs/v3_video.yaml --data data/v3_video \
  --gpus 1 --num-workers 4 --blacklist data/v3_video/_qa/blacklist.txt

# Eval
python scripts/eval_v3.py --checkpoint checkpoints_v3_video/epoch_0050.pt \
  --data data/v3_video --max-windows 2000 --report eval_report_v3_video.md

# CEM open-loop (optionnel)
python scripts/plan_v3.py --checkpoint checkpoints_v3_video/epoch_0050.pt \
  --data data/v3_video --n-pairs 100 --horizon 8 --report plan_report_v3.md
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
