# Rapport CEM open-loop V3 (inverse planning)

- Checkpoint: `checkpoints_v3_lambda30/epoch_0030.pt`
- Config: `configs/v3_video_lambda30.yaml`
- Dataset: `data/v3_video`
- Device: `cuda`
- Date: `2026-05-08T16:45:28`

## Paramètres

- Pairs: `100`  (windows tirées au hasard du val set)
- Horizon: `8`
- CEM samples / iters / elite: `512` / `4` / `64`

## Résultats

- MSE initial moyen (z_0 vs z_T_target avant CEM): `14.889066`
- MSE final moyen (après CEM): `6.953726`
- **Ratio MSE_final / MSE_initial** moyen: `1.306003`
- Ratio médian: `0.477495`

## Interprétation

ratio < 0.5 = dynamics inversible utile pour planning ; ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation.
