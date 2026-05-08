# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v4_2_lambdacov15/epoch_0030.pt`
- Config: `configs/v4_2_lambdacov15.yaml`
- Dataset: `LunarLanderContinuous-v3 (heuristic, 100 ep)`
- Device: `cuda`
- Date: `2026-05-08T23:50:37`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.000731 | 619 |
| 2 | 0.000698 | 619 |
| 4 | 0.000688 | 619 |
| 8 | 0.000731 | 619 |
| 16 | 0.000906 | 619 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.238942`

## Statistiques latents

- Variance moyenne / dim : `0.016068`
- Variance min / dim : `0.000136`
- Variance max / dim : `0.082213`
- **Effective rank** : `2.12` / `256` (`0.83%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.000731 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.238942 |
| Effective rank | 20.6/256 (8%) | 2.1/256 (0.8%) |

## Interprétation

Rank 2.1/256 → le modèle utilise 0.8% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
