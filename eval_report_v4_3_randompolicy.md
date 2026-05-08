# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v4_3_randompolicy/epoch_0030.pt`
- Config: `configs/v4_2_lambdacov15.yaml`
- Dataset: `LunarLanderContinuous-v3 (random, 100 ep)`
- Device: `cuda`
- Date: `2026-05-09T00:26:44`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.002620 | 427 |
| 2 | 0.002323 | 427 |
| 4 | 0.002530 | 427 |
| 8 | 0.003126 | 427 |
| 16 | 0.004380 | 427 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.671727`

## Statistiques latents

- Variance moyenne / dim : `0.030204`
- Variance min / dim : `0.000326`
- Variance max / dim : `0.168330`
- **Effective rank** : `2.43` / `256` (`0.95%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.002620 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.671727 |
| Effective rank | 20.6/256 (8%) | 2.4/256 (1.0%) |

## Interprétation

Rank 2.4/256 → le modèle utilise 1.0% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
