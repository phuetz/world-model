# Rapport CEM open-loop V3 (inverse planning)

- Checkpoint: `checkpoints_v3_partial/epoch_0010.pt`
- Config: `configs/v3_video.yaml`
- Dataset: `data/v3_video`
- Device: `cuda`
- Date: `2026-05-01T12:55:25`

## Paramètres

- Pairs: `10`  (windows tirées au hasard du val set)
- Horizon: `8`
- CEM samples / iters / elite: `256` / `4` / `32`

## Résultats

- MSE initial moyen (z_0 vs z_T_target avant CEM): `0.002075`
- MSE final moyen (après CEM): `0.006797`
- **Ratio MSE_final / MSE_initial** moyen: `20.149483`
- Ratio médian: `11.173707`

## Interprétation

ratio < 0.5 = dynamics inversible utile pour planning ; ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation.
