# Rapport CEM open-loop V3 (inverse planning)

- Checkpoint: `checkpoints_v3_1_wan/epoch_0030.pt`
- Config: `configs/v3_video.yaml`
- Dataset: `data/v3_video_wan`
- Device: `cuda`
- Date: `2026-05-08T16:46:50`

## Paramètres

- Pairs: `70`  (windows tirées au hasard du val set)
- Horizon: `8`
- CEM samples / iters / elite: `512` / `4` / `64`

## Résultats

- MSE initial moyen (z_0 vs z_T_target avant CEM): `0.002548`
- MSE final moyen (après CEM): `0.001597`
- **Ratio MSE_final / MSE_initial** moyen: `19.789263`
- Ratio médian: `1.254820`

## Interprétation

ratio < 0.5 = dynamics inversible utile pour planning ; ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation.
