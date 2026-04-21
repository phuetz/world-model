# Rapport de planning World Model

- Checkpoint: `checkpoints_carracing_rollout5/epoch_0100.pt`
- Env: `CarRacing-v3`
- Episodes: `3` × `400` steps max
- CEM: horizon=12, samples=512, elites=64, iters=4
- Date: `2026-04-22T01:15:38`

## Résultats par politique

| Politique | Return moyen | Return médian | Length moyenne |
|---|---:|---:|---:|
| random | -7.46 | -7.51 | 400 |
| heuristic | -22.54 | -22.33 | 400 |
| cem | -6.32 | -3.90 | 400 |
