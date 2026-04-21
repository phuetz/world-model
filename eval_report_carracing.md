# Rapport d'evaluation World Model JEPA

- Checkpoint: `checkpoints_carracing/epoch_0100.pt`
- Env: `CarRacing-v3`
- Samples: `2000`
- Device: `cuda`
- Date: `2026-04-21T20:42:59`

## MSE par horizon

| Horizon | MSE latent | Points valides |
|---:|---:|---:|
| 1 | 0.008672 | 2000 |
| 5 | 0.042215 | 1990 |
| 10 | 0.141220 | 1980 |
| 20 | 119.681618 | 1960 |

## Statistiques du latent

- Variance moyenne par dimension: `0.075566`
- Variance min par dimension: `0.008044`
- Variance max par dimension: `0.276855`
- Effective rank: `14.72` / `256` (`5.75%`)

## Interpretation

Rank 14.7/256 -> le modele utilise 5.8% de sa capacite latente.
