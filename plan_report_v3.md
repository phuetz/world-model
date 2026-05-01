# Rapport CEM open-loop V3 (inverse planning)

- Checkpoint: `checkpoints_v3_video\epoch_0030.pt`
- Config: `configs/v3_video.yaml`
- Dataset: `data\v3_video`
- Device: `cuda`
- Date: `2026-05-01T18:30:38`

## Paramètres

- Pairs: `100`  (windows tirées au hasard du val set)
- Horizon: `8`
- CEM samples / iters / elite: `512` / `4` / `64`

## Résultats

- MSE initial moyen (z_0 vs z_T_target avant CEM): `0.030847`
- MSE final moyen (après CEM): `0.019477`
- **Ratio MSE_final / MSE_initial** moyen: `1.221845`
- Ratio médian: `0.878079`

## Interprétation

ratio < 0.5 = dynamics inversible utile pour planning ; ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation.
