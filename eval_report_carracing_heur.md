# Rapport d'evaluation World Model JEPA

- Checkpoint: `checkpoints_carracing_heur/epoch_0100.pt`
- Env: `CarRacing-v3`
- Samples: `2000`
- Device: `cuda`
- Date: `2026-04-21T23:37:21`

## MSE par horizon

| Horizon | MSE latent | Points valides |
|---:|---:|---:|
| 1 | 0.038023 | 2000 |
| 5 | 0.163735 | 1990 |
| 10 | 0.198311 | 1980 |
| 20 | 0.169958 | 1960 |

## Statistiques du latent

- Variance moyenne par dimension: `0.059181`
- Variance min par dimension: `0.005931`
- Variance max par dimension: `0.662016`
- Effective rank: `13.67` / `256` (`5.34%`)

## Interpretation

Rank 13.7/256 -> le modele utilise 5.3% de sa capacite latente.
