# Synthèse V4 — Gymnasium real-env (LunarLanderContinuous)

> Daté 2026-05-08. Suite de `eval_synthesis_v3.md`.
> V4 a été lancé pour tester l'hypothèse "actions vraies > optical flow proxy".
> Résultat : **hypothèse réfutée**. Le rank collapse persiste sur env Gymnasium.

## Hypothèse testée

V1.8 sur CarRacing (MLP 2.5M + actions vraies) atteignait rank 8 %. V3
sur SVD-XT (Conv5 + Transformer + optical flow) plafonne à 2.9 %.
Conclusion provisoire de la synthèse V3.x : **l'optical flow proxy est
le verrou**. Pivot V4 : reprendre l'archi V3 (Conv4 + Transformer) sur
Gymnasium real-env pour combiner capacité Transformer + signal action vrai.

## Setup V4

- **Env** : `LunarLanderContinuous-v3`, render_mode `rgb_array` 400×600 → resized 64×64
- **Policy** : heuristic (main engine + lateral oscillation, `gym_video_dataset.py`)
- **Dataset** : 100 episodes × max 200 steps, 5603 windows totales (5007 train / 596 val)
- **Archi** : Conv4 (V1.8) + Transformer dynamique 4×8×512 (V3), latent_dim 256, **5.2M params**
- **Hyperparams** : lr 5e-5, batch 32, 30 epochs, λ_var=0.04, fp32, warmup_epochs=10
- **Run #1 abandonné** (lr 1e-4 + warmup 5) : divergé epoch 3 (loss_pred 0.06 → 97 → 8).
  Conv4 64×64 sur LunarLander capture peu, latents fluctuent fort, lr 1e-4 trop haut.
  Recul à V3-stable : lr 5e-5 + warmup 10 → run #2 stable et convergé.

## Résultats V4 (run #2, epoch 30)

| Horizon | MSE latent | Windows |
|---:|---:|---:|
| 1 | 0.000233 | 619 |
| 2 | 0.000228 | 619 |
| 4 | 0.000238 | 619 |
| 8 | 0.000292 | 619 |
| 16 | 0.000506 | 619 |

- **Compounding ratio MSE(h=16) / MSE(h=1) : ×2.17**
- Variance moyenne / dim : 0.0153
- **Effective rank : 2.35 / 256 (0.9 %)**

**Plan V4 (CEM open-loop, 100 paires, horizon 8) :**
- MSE initial moyen (z_0 vs z_T_target) : **0.000001** — quasi nul, signe que
  les latents tirés au hasard sont pratiquement identiques
- MSE final moyen (après CEM) : 0.000300 — CEM **aggrave** la distance ×300
- Ratios moyen/médian `n/a` (division par zéro : init ≈ 0)

C'est la signature classique du **latent collapse** : tous les états latents
vivent dans un sous-espace minuscule, MSE est trivialement bas, et tout
mouvement (y compris ce que CEM trouve de "meilleur") sort de ce sous-espace
et augmente la distance. CEM = inutilisable pour planning.

## Tableau comparatif consolidé

| Run | Dataset | Encoder | Dynamics | λ_var | Params | MSE h=1 | Compounding | **Rank /dim** |
|---|---|---|---|---:|---:|---:|---:|---:|
| **V1.8** | CarRacing actions vraies | Conv4 | MLP | 0.04 | 2.5M | 0.0135 | ×2.8 | **20.6/256 (8.0 %)** |
| V3 | SVD-XT 1500 clips | Conv5 | Transformer | 0.15 | 23.8M | 0.0178 | ×1.55 | 14.7/512 (2.9 %) |
| V3.0.1 | SVD-XT 1500 (λ_var=0.30) | Conv5 | Transformer | 0.30 | 23.8M | 6.5625 | ×1.20 | 7.2/512 (1.4 %) |
| V3.1 | Wan 2.2 i2v 300 clips | Conv5 | Transformer | 0.15 | 23.8M | 0.0020 | ×1.84 | 1.4/512 (0.3 %) |
| **V4** | LunarLanderContinuous heuristic | Conv4 | Transformer | 0.04 | 5.2M | **0.0002** | ×2.17 | **2.4/256 (0.9 %)** |

## Lecture

### Ce qui se confirme
- **Le pipeline V4 fonctionne end-to-end** : collecte env Gymnasium, training,
  eval/plan avec backend gym. Code réutilisable (cf. `gym_video_dataset.py`,
  `train_v4.py`, eval/plan `--backend gym`).
- **MSE h=1 record absolu : 0.000233** — V4 prédit la transition mieux qu'aucun
  prédécesseur. Cohérent avec l'idée que les actions vraies (vs optical flow)
  donnent un signal causal exploitable.

### Ce qui réfute l'hypothèse
**L'effective rank ne remonte pas avec les actions vraies seules.** V4 plafonne
à 0.9 %, plus bas que V3 SVD-XT (2.9 %) et **9× plus bas que V1.8 (8 %)**. Le
verrou n'est donc pas l'optical flow proxy en soi.

### Diagnostic révisé

Le rank collapse semble venir de la **combinaison Transformer dynamique
+ VICReg λ_var trop faible**, indépendamment du dataset :

- V1.8 utilisait un **MLP 2-layer** comme dynamique. Capacité limitée → forcé
  d'utiliser plusieurs dimensions latentes pour fitter les transitions.
- V3 / V3.1 / V4 utilisent un **Transformer 4×8×512**. Capacité énorme →
  peut fitter MSE basse avec un sous-espace latent minimal (1-2 dims actives,
  reste à variance ~0). VICReg à λ_var=0.04 (V1.8 setting) ne pousse pas
  assez.

V3.0.1 (λ_var=0.30) a confirmé qu'un λ trop fort casse la prédiction.
V3 (λ_var=0.15) tient le compromis MSE/rank le plus haut (2.9 %), mais
même là on est sous la cible 15 %.

## Recommandations

### Court terme (à tester en priorité)

1. **V4.1 — λ_var=0.15** sur LunarLander, même archi V4. Test le plus direct :
   λ_var=0.15 sur env vrai donne-t-il rank > V3 (2.9 %) ? Si oui, l'hypothèse
   "λ_var=0.04 trop faible avec Transformer" est validée.
2. **V4.2 — random policy** au lieu de heuristic. La politique heuristique
   sinusoïdale produit des trajectoires très similaires d'un episode à
   l'autre → diversité limitée. Random sample plus large.
3. **Re-test V1.8 archi (MLP) sur LunarLander** : si MLP donne rank ≥ 8 %
   sur LunarLander aussi, on confirme que c'est le Transformer dynamique
   qui collapse, pas le dataset.

### Moyen terme

- **VICReg revisité** : implémenter le terme `var_loss` avec target std=1.0
  vérifié bien (au lieu d'une simple penalty), ou tester DCorr/Barlow Twins
  comme alternative.
- **CarRacing-v3 + Transformer** : env plus riche (3D actions vs 2D) avec
  notre archi V4 — voir si le rank monte naturellement avec plus de
  dimension d'action.
- **Latent_dim plus petit** (128 au lieu de 256/512) : si la capacité latente
  excessive est le problème, réduire force le modèle à utiliser ses dims.

## Artefacts produits cette session (suite 2026-05-08)

- Squelette V4 : `gym_video_dataset.py`, `configs/v4_lunarlander.yaml`,
  `scripts/train_v4.py` (commit `03fe1de`)
- Eval/plan extension `--backend gym` : `eval_v3.py`, `plan_v3.py`
  (commit `53070b7`)
- CLAUDE.md V4 doc (commit `3dbe2a0`)
- Eval V4 : `eval_report_v4_lunarlander.md`
- Plan V4 : `plan_report_v4_lunarlander.md`
- Cette synthèse

— Claude / DARKSTAR, 2026-05-08
