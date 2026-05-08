# Rapport CEM open-loop V3 (inverse planning)

- Checkpoint: `checkpoints_v4_1_lambda15/epoch_0030.pt`
- Config: `configs/v4_1_lambda15.yaml`
- Dataset: `LunarLanderContinuous-v3 (heuristic, 100 ep)`
- Device: `cuda`
- Date: `2026-05-08T19:30:49`

## Paramètres

- Pairs: `100`  (windows tirées au hasard du val set)
- Horizon: `8`
- CEM samples / iters / elite: `512` / `4` / `64`

## Résultats

- MSE initial moyen (z_0 vs z_T_target avant CEM): `0.000001`
- MSE final moyen (après CEM): `0.000650`
- **Ratio MSE_final / MSE_initial** moyen: `n/a`
- Ratio médian: `n/a`

## Interprétation

ratio < 0.5 = dynamics inversible utile pour planning ; ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation.
