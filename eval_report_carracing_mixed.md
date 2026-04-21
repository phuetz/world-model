# Rapport d'evaluation World Model JEPA

- Checkpoint: `checkpoints_carracing_mixed/epoch_0100.pt`
- Env: `CarRacing-v3`
- Samples: `2000`
- Device: `cuda`
- Date: `2026-04-21T23:55:47`

## MSE par horizon

| Horizon | MSE latent | Points valides |
|---:|---:|---:|
| 1 | 0.030228 | 2000 |
| 5 | 0.103480 | 1990 |
| 10 | 0.159571 | 1980 |
| 20 | 0.229488 | 1960 |

## Statistiques du latent

- Variance moyenne par dimension: `0.149949`
- Variance min par dimension: `0.018104`
- Variance max par dimension: `0.629117`
- Effective rank: `23.06` / `256` (`9.01%`)

## Interpretation

Rank 23.1/256 -> le modele utilise 9.0% de sa capacite latente.
