# Synthèse V3.x — comparatif des configurations

> Daté 2026-05-08. Compile les rapports `eval_report_v3*.md` et `plan_report_v3*.md`
> produits sur DARKSTAR (2× RTX 3090). Tous les checkpoints sont à `epoch_0030.pt`.

## Tableau comparatif

| Run | Dataset | clips | λ_var | MSE h=1 | Compounding (h=16/h=1) | Effective rank /512 | CEM ratio méd. |
|---|---|---:|---:|---:|---:|---:|---:|
| **V1.8 (baseline)** | CarRacing teacher-forced | n/a | 0.04 | 0.0135 | ×2.8 (h=20) | 20.6/256 (8.0%) | n/a |
| V3 partial | SVD-XT | 51 | 0.15 | 0.0070 | ×1.54 | 1.51/512 (0.3%) | 11.17 (dégradé) |
| **V3** | SVD-XT | 1500 | 0.15 | **0.0178** | **×1.55** | **14.70/512 (2.9%)** | **0.88 (utile)** |
| V3.0.1 | SVD-XT | 1500 | **0.30** | 6.5625 | ×1.20 | 7.19/512 (1.4%) | 0.48 |
| **V3.1 Wan** | Wan 2.2 i2v | 300 | 0.15 | **0.0020** | ×1.84 | **1.41/512 (0.3%)** | 1.25 (peu utile) |

## Lecture

### Ce qui marche
- **V3 sur SVD-XT 1500 clips reste la meilleure config globale.** MSE acceptable, compounding sous cible 2.0, rank 2.9 % (sous cible 15 % mais pas effondré), CEM open-loop utile (ratio méd. 0.88 → la dynamique est inversible sur la majorité des paires).
- V3.1 Wan donne le **meilleur MSE absolu (×9 vs V3)** : Wan 2.2 produit des clips photo-réalistes que le modèle prédit très bien.

### Ce qui ne marche pas
- **V3.0.1 (λ_var=0.30) casse la prédiction.** Variance latente moyenne explose (7.98 vs 0.04 en V3), MSE × 360 pire que V3. La régularisation a pris le pas sur la loss prédictive. **Conclusion : λ_var ∈ [0.04, 0.15] est la bonne plage, 0.30 est trop fort.**
- **V3.1 Wan a un latent collapse complet** : rank 0.28 % (pire que V3 SVD-XT, et 30× plus pauvre que V1.8). Le modèle prédit l'identité parce que les clips Wan ont très peu de mouvement inter-frame. CEM ne fonctionne plus (ratio méd. 1.25 → dégrade sur la majorité des paires).
- V3 partial (51 clips, epoch 10) a déjà rank 0.3 % : le rank ne monte qu'avec **plus de données ET plus d'epochs**. Ce n'est pas qu'un effet de sous-entraînement.

## Diagnostic

Les trois axes (architecture, hyperparams, dataset) sont impliqués différemment :

1. **Architecture** : Conv5 + Transformer 4×8×512 fonctionne (V3 atteint 2.9 % rank, compounding 1.55 — supérieur à V1.8 sur ce dernier critère). Pas de raison de toucher.

2. **Hyperparams VICReg** : λ_var=0.15 est calibré. Plus fort (0.30) casse la prédiction, plus faible (0.04 V1.8) aurait probablement aggravé le rank sur dataset vidéo. Ce n'est **pas** ici qu'il faut chercher des gains.

3. **Dataset** — c'est le verrou.
   - SVD-XT et Wan 2.2 produisent du **i2v passif** (image → animation cohérente). L'action proxy 4D dérivée de Farneback optical flow est **dégénérée sur scènes statiques** : peu de signal exploitable.
   - Wan, plus stable visuellement, accentue le problème → modèle qui prédit l'identité, latent collapse.
   - Pour faire monter le rank il faut soit (a) un dataset i2v avec **plus de mouvement réel** (à explorer : prompts avec verbes d'action, vidéos plus longues, objets mobiles), soit (b) **abandonner le dataset vidéo passif** et passer à un environnement avec actions discrètes contrôlables.

## Recommandation pour V4

**Pivot vers Gymnasium real-env** comme prévu dans la roadmap initiale (cf. CLAUDE.md "V4 : modalité audio, Gymnasium réel LunarLander/Pusher, ONNX export").

Justification : V1.8 sur CarRacing avec actions vraies a atteint 8 % rank sans effort spécifique. Sur des envs avec espaces d'action plus riches (LunarLander = continuous 2D, Pusher = continuous 7D), le signal d'action contraint le latent à ne pas s'effondrer. C'est aussi cohérent avec l'objectif robot final : un robot a un espace d'action discret/contrôlé, pas un signal optical flow.

### Plan V4 minimal proposé
1. `world_model.data.gym_replay` — buffer de transitions (obs, action, next_obs) sur LunarLander-v3 (action dim 2 continuous) ou PusherEnv (7D).
2. Adapter Conv5 encoder en gardant 84×84 ou 256×256 selon env.
3. Reprendre Transformer dynamique (déjà éprouvé V3) avec actions vraies au lieu d'optical flow.
4. Cibles : MSE h=1 < 0.02, compounding < 2.0, **rank ≥ 10 %** (objectif minimum, V1.8 est à 8 % donc on doit faire mieux avec latent_dim=512).

### Alternative (plus rapide, plus risquée)
**V3.2 — dataset Wan 2.2 enrichi en mouvement** : régénérer 1000 clips avec prompts orientés action explicite (`hand throwing ball`, `runner sprinting`, `door slamming shut`...) plutôt que prompts statiques. Garder l'archi V3 telle quelle. Permet de garder l'investissement vidéo si le but est l'usage robot end-to-end (vision passive). Risque : si l'action proxy reste pauvre, latent collapse persiste.

## Artefacts produits cette session (2026-05-08)

- Eval/plan **V3.0.1** (jamais évalué auparavant) : `eval_report_v3_lambda30.md`, `plan_report_v3_lambda30.md`
- Eval/plan **V3.1 Wan** (jamais commit/push avant) : `eval_report_v3_1_wan.md`, `plan_report_v3_1_wan.md`
- Synthèse comparative : ce fichier
- Wrap-up orchestrateur : `scripts/wrap_up_v3.py` (nouveau)

— Claude / DARKSTAR, 2026-05-08
