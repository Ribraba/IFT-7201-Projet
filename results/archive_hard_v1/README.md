# Archive — hard map v1 (carte initiale)

## Carte
```
SFFFFF
HFHFFF
FFHFHF
FHFFHF
FFHFFF
FFFFHG
```
- 6×6, is_slippery=True, 9 trous

## Résultats (10 runs, 500 épisodes d'évaluation)
- Q-learning : succès ~7.6%, chutes ~91.6%
- SARSA      : succès ~1.7%, chutes ~98.0%

## Pourquoi redesignée
Taux de chutes trop élevé (~91-98%) — les deux algorithmes échouent massivement,
impossible d'observer des comportements différenciés ou une progression pendant
l'apprentissage. La carte a été redesignée (v2) pour être difficile mais apprennable,
permettant une comparaison significative entre Q-learning et SARSA.
