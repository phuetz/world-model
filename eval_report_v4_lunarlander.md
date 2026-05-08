# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v4_lunarlander/epoch_0030.pt`
- Config: `configs/v4_lunarlander.yaml`
- Dataset: `LunarLanderContinuous-v3 (heuristic, 100 ep)`
- Device: `cuda`
- Date: `2026-05-08T18:11:23`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.000233 | 619 |
| 2 | 0.000228 | 619 |
| 4 | 0.000238 | 619 |
| 8 | 0.000292 | 619 |
| 16 | 0.000506 | 619 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `2.173246`

## Statistiques latents

- Variance moyenne / dim : `0.015344`
- Variance min / dim : `0.000152`
- Variance max / dim : `0.076710`
- **Effective rank** : `2.35` / `256` (`0.92%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.000233 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 2.173246 |
| Effective rank | 20.6/256 (8%) | 2.3/256 (0.9%) |

## Interprétation

Rank 2.3/256 → le modèle utilise 0.9% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
