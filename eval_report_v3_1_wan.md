# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v3_1_wan/epoch_0030.pt`
- Config: `configs/v3_video.yaml`
- Dataset: `data/v3_video_wan`
- Device: `cuda`
- Date: `2026-05-08T16:46:31`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.001968 | 70 |
| 2 | 0.002071 | 70 |
| 4 | 0.002612 | 70 |
| 8 | 0.003066 | 70 |
| 16 | 0.003629 | 70 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.843683`

## Statistiques latents

- Variance moyenne / dim : `0.020309`
- Variance min / dim : `0.000044`
- Variance max / dim : `0.268895`
- **Effective rank** : `1.41` / `512` (`0.28%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.001968 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.843683 |
| Effective rank | 20.6/256 (8%) | 1.4/512 (0.3%) |

## Interprétation

Rank 1.4/512 → le modèle utilise 0.3% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
