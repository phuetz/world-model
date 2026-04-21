# Rapport d'evaluation World Model JEPA

- Checkpoint: `checkpoints_carracing_rollout5/epoch_0100.pt`
- Env: `CarRacing-v3`
- Samples: `2000`
- Device: `cuda`
- Date: `2026-04-22T00:30:07`

## MSE par horizon

| Horizon | MSE latent | Points valides |
|---:|---:|---:|
| 1 | 0.013486 | 2000 |
| 5 | 0.020974 | 1990 |
| 10 | 0.027927 | 1980 |
| 20 | 0.037630 | 1960 |

## Statistiques du latent

- Variance moyenne par dimension: `0.025390`
- Variance min par dimension: `0.006515`
- Variance max par dimension: `0.069712`
- Effective rank: `20.61` / `256` (`8.05%`)

## Interpretation

Rank 20.6/256 -> le modele utilise 8.1% de sa capacite latente.
