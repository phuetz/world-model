# Rapport d'evaluation World Model JEPA

- Checkpoint: `checkpoints_carracing_hv/epoch_0100.pt`
- Env: `CarRacing-v3`
- Samples: `2000`
- Device: `cuda`
- Date: `2026-04-21T23:16:43`

## MSE par horizon

| Horizon | MSE latent | Points valides |
|---:|---:|---:|
| 1 | 0.013247 | 2000 |
| 5 | 0.080288 | 1990 |
| 10 | 7.152852 | 1980 |
| 20 | 27318536.000000 | 1960 |

## Statistiques du latent

- Variance moyenne par dimension: `0.133760`
- Variance min par dimension: `0.013288`
- Variance max par dimension: `0.361213`
- Effective rank: `16.21` / `256` (`6.33%`)

## Interpretation

Rank 16.2/256 -> le modele utilise 6.3% de sa capacite latente.
