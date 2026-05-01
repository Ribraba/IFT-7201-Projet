# IFT-7201 — Projet : Apprentissage par renforcement sur FrozenLake

Étude comparative de Q-learning, SARSA et d'une variante de Q-learning blindé sur trois variantes de l'environnement FrozenLake-v1 (Gymnasium).

## Structure du projet

```
├── config/
│   └── config_baselines.yaml   # Hyperparamètres et paramètres expérimentaux
├── src/
│   ├── agents.py               # Q-learning, SARSA, Q-learning blindé
│   ├── envs.py                 # Environnements (Facile, Moyen, Difficile)
│   ├── evaluate.py             # Évaluation greedy, avec/sans shield, + analyse Dijkstra
│   └── plots.py                # Figures (courbes, barres, heatmap, boîtes à moustaches)
├── run_experiments.py          # Script principal d'entraînement
├── FrozenLakeProject.ipynb     # Vitrine : visualisation des environnements et résultats
├── results/                    # Résultats JSON par expérience/run (généré)
├── figures/                    # Figures PNG (générées)
└── docs/                   # Rapports LaTeX (analyse préliminaire, finale...) ainsi que les autres documents nécessaires au projet
    └── ...
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduire les expériences

```bash
python run_experiments.py --config config/config_baselines.yaml --results results
```

Les résultats sont sauvegardés dans `results/<nom_experience>/run_<i>.json`. Les runs déjà calculés sont ignorés (reprise possible en cas d'interruption).

Pour `shielded_qlearning`, chaque JSON contient deux évaluations distinctes :
- `evaluation` : politique greedy finale **sans shield** au déploiement
- `evaluation_with_shield` : même table Q, mais politique greedy **avec shield maintenu** à l'inférence

## Visualiser les résultats

Ouvrir `FrozenLakeProject.ipynb` dans Jupyter. Le notebook charge les résultats depuis `results/` et génère toutes les figures dans `figures/`.

Une figure ciblée supplémentaire met en évidence le compromis sécurité/performance sur le cas `hard` :
- `figures/hard_shield_tradeoff.png` compare `Q-learning`, `SARSA`, `Q-learning blindé sans shield` et `Q-learning blindé avec shield` sur les métriques de succès, chutes et boucles

## Archives

| Dossier | Contenu |
|---|---|
| `results/archive/archive_hard_v1/` | Première version de la carte Difficile (falaise trop large, taux de chutes > 90% pour les deux algorithmes — comparaison non informative). Redessinée en s'inspirant du *cliff walking* Sutton & Barto. |
| `results/archive/archive_khalil_maps/` | Cartes alternatives testées avec les mêmes hyperparamètres (20 000 épisodes, 10 runs, pas de cherry-picking). Résultat : Δ succès = 0.4% et Δ chemin sûr = 1.6% sur la carte Difficile, contre respectivement 12.2% et 10.3% pour les cartes principales. Cartes principales conservées. |

## Paramètres expérimentaux

Voir `config/config_baselines.yaml` pour le détail complet. Résumé :

| Paramètre | Valeur |
|---|---|
| Épisodes d'entraînement | 20 000 |
| Pas max par épisode | 200 |
| Répétitions (graines) | 10 (0 à 9) |
| Épisodes d'évaluation | 500 |
| γ | 0.99 |
| α₀ / αₘᵢₙ | 0.5 / 0.01 |
| ε₀ / εₘᵢₙ | 1.0 / 0.05 |
| Décroissance α et ε | 0.9995 |

## Note méthodologique

- La différence `sans shield` / `avec shield` concerne uniquement l'évaluation finale : l'entraînement et la table Q sont identiques, seul le mécanisme de sélection d'action à l'inférence change.
