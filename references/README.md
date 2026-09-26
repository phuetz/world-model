# Références

## LeWorldModel (arXiv 2603.19312)

**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**
Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero — mars 2026.
<https://arxiv.org/abs/2603.19312>

Fichier : `2603.19312-LeWorldModel.pdf`, redistribué sans modification sous licence
[Creative Commons Attribution 4.0 (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).

Pourquoi ici : un JEPA qui apprend de façon stable depuis les pixels avec seulement deux termes de coût
(prédiction du prochain plongement + régulariseur imposant des latents gaussiens), ~15 M de paramètres,
entraînable en quelques heures sur un GPU, avec détection de la « surprise ». À comparer au régulariseur
isotrope de ce dépôt (V1 à V2.0, CarRacing).
