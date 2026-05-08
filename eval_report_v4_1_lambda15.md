# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v4_1_lambda15/epoch_0030.pt`
- Config: `configs/v4_1_lambda15.yaml`
- Dataset: `LunarLanderContinuous-v3 (heuristic, 100 ep)`
- Device: `cuda`
- Date: `2026-05-08T19:29:35`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.000695 | 619 |
| 2 | 0.000688 | 619 |
| 4 | 0.000693 | 619 |
| 8 | 0.000711 | 619 |
| 16 | 0.000780 | 619 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.122344`

## Statistiques latents

- Variance moyenne / dim : `0.035523`
- Variance min / dim : `0.002049`
- Variance max / dim : `0.254951`
- **Effective rank** : `2.28` / `256` (`0.89%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.000695 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.122344 |
| Effective rank | 20.6/256 (8%) | 2.3/256 (0.9%) |

## Interprétation

Rank 2.3/256 → le modèle utilise 0.9% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
