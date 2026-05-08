# Rapport CEM open-loop V3 (inverse planning)

- Checkpoint: `checkpoints_v4_3_randompolicy/epoch_0030.pt`
- Config: `configs/v4_2_lambdacov15.yaml`
- Dataset: `LunarLanderContinuous-v3 (random, 100 ep)`
- Device: `cuda`
- Date: `2026-05-09T00:27:57`

## Paramètres

- Pairs: `100`  (windows tirées au hasard du val set)
- Horizon: `8`
- CEM samples / iters / elite: `512` / `4` / `64`

## Résultats

- MSE initial moyen (z_0 vs z_T_target avant CEM): `0.000012`
- MSE final moyen (après CEM): `0.001154`
- **Ratio MSE_final / MSE_initial** moyen: `n/a`
- Ratio médian: `n/a`

## Interprétation

ratio < 0.5 = dynamics inversible utile pour planning ; ratio ~ 1.0 = pas de gain CEM ; ratio > 1.0 = dégradation.
