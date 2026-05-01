# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v3_video\epoch_0030.pt`
- Config: `configs/v3_video.yaml`
- Dataset: `data\v3_video`
- Device: `cuda`
- Date: `2026-05-01T18:30:24`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.017764 | 375 |
| 2 | 0.021657 | 375 |
| 4 | 0.023744 | 375 |
| 8 | 0.023006 | 375 |
| 16 | 0.027492 | 375 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.547641`

## Statistiques latents

- Variance moyenne / dim : `0.040212`
- Variance min / dim : `0.001436`
- Variance max / dim : `0.266615`
- **Effective rank** : `14.70` / `512` (`2.87%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.017764 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.547641 |
| Effective rank | 20.6/256 (8%) | 14.7/512 (2.9%) |

## Interprétation

Rank 14.7/512 → le modèle utilise 2.9% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
