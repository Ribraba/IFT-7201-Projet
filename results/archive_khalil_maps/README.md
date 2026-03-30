# Archive — Cartes alternatives (branche second_version) testées avec l'entraînement principal

Expériences réalisées le 2026-03-30 : cartes de la branche second_version + hyperparamètres du rapport principal (20 000 épisodes, 10 runs, pas de cherry-picking).

## Cartes testées
- **easy_k / medium_k** : 6×6, `["SFFFFG","FFHFHF","FFFGHF","FFHFHF","FFFFFH","FFFFFG"]`, 3 G, 6 trous
- **hard_k** : 6×6, `["SFFFFF","FFHFHF","FFHGFF","FFFFFH","FFFFFF","HHFHHH"]`, 1 G central, 9 trous

## Résultats

| Expérience             | Succès          | Chutes          | Boucles | Chemin sûr      |
|------------------------|-----------------|-----------------|---------|-----------------|
| qlearning_easy_k       | 100.0 ± 0.0%   | 0.0 ± 0.0%     | 0.0%    | N/A             |
| sarsa_easy_k           | 100.0 ± 0.0%   | 0.0 ± 0.0%     | 0.0%    | N/A             |
| qlearning_medium_k     | 72.5 ± 2.3%    | 8.9 ± 3.8%     | 18.7%   | N/A             |
| sarsa_medium_k         | 66.8 ± 3.8%    | 31.8 ± 3.9%    | 1.4%    | N/A             |
| qlearning_hard_k       | 67.2 ± 1.9%    | 24.9 ± 0.8%    | 8.0%    | 47.8 ± 2.4%    |
| sarsa_hard_k           | 66.7 ± 3.3%    | 26.4 ± 4.0%    | 6.9%    | 49.4 ± 2.8%    |

## Problèmes identifiés

| Environnement | Problème |
|---|---|
| **easy_k** | Effet plafond : 100% succès pour QL et SARSA → aucune divergence mesurable |
| **medium_k** | Résultats inversés : QL > SARSA en succès, SARSA chute 3× plus → incompatible avec la théorie cliff-walking |
| **hard_k** | Divergence quasi nulle : Δ succès = −0.4%, Δ chemin sûr = +1.6% (non significatif) |

**Raison structurelle** : le but G est au centre de la grille entouré de trous — il n'existe pas de dichotomie chemin risqué / chemin sûr. Sans cette structure, les deux algorithmes adoptent des comportements quasi-identiques.

## Comparaison avec les cartes du rapport final (hard)

| Métrique | Cartes second_version (hard_k) | Cartes rapport final (hard) |
|---|---|---|
| Δ Succès (SARSA − QL) | **−0.4%** | **+12.2%** |
| Δ Boucles (SARSA − QL) | **−1.1%** | **−12.2%** |
| Δ Chemin sûr (SARSA − QL) | **+1.6%** | **+10.3%** |

→ **Les cartes du rapport final sont conservées** : elles seules produisent la divergence comportementale attendue entre Q-learning et SARSA.
