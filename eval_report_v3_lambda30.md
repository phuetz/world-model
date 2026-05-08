# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v3_lambda30/epoch_0030.pt`
- Config: `configs/v3_video_lambda30.yaml`
- Dataset: `data/v3_video`
- Device: `cuda`
- Date: `2026-05-08T16:44:33`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 6.562506 | 375 |
| 2 | 7.593927 | 375 |
| 4 | 7.778677 | 375 |
| 8 | 7.610984 | 375 |
| 16 | 7.851648 | 375 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.196441`

## Statistiques latents

- Variance moyenne / dim : `7.977842`
- Variance min / dim : `0.314988`
- Variance max / dim : `47.841114`
- **Effective rank** : `7.19` / `512` (`1.41%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 6.562506 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.196441 |
| Effective rank | 20.6/256 (8%) | 7.2/512 (1.4%) |

## Interprétation

Rank 7.2/512 → le modèle utilise 1.4% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
