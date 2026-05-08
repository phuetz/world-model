# Rapport d'évaluation V3 (video)

- Checkpoint: `checkpoints_v3_partial/epoch_0010.pt`
- Config: `configs/v3_video.yaml`
- Dataset: `data/v3_video`
- Device: `cuda`
- Date: `2026-05-01T12:54:28`

## MSE multi-step

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.007004 | 15 |
| 2 | 0.011410 | 15 |
| 4 | 0.011785 | 15 |
| 8 | 0.010596 | 15 |
| 16 | 0.010783 | 15 |

- **Compounding ratio MSE(h=16) / MSE(h=1)** : `1.539629`

## Statistiques latents

- Variance moyenne / dim : `0.000852`
- Variance min / dim : `0.000022`
- Variance max / dim : `0.005487`
- **Effective rank** : `1.51` / `512` (`0.30%`)

## Comparaison V1.8 baseline (CarRacing)

| Métrique | V1.8 | V3 |
|---|---|---|
| MSE h=1 | 0.0135 | 0.007004 |
| Compounding (V1.8 h=20, V3 h=16) | x2.8 | 1.539629 |
| Effective rank | 20.6/256 (8%) | 1.5/512 (0.3%) |

## Interprétation

Rank 1.5/512 → le modèle utilise 0.3% de sa capacité latente. Cible >15% = succès architectural ; si <10% → re-tune lambda_var en V3.0.1.
