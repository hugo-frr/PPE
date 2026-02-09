# Journal d'Evolution du Projet

## Entree 1 - Cadrage technique

- Decision: centraliser les parametres dans `src/config.py`.
- Raison: eviter les chemins hardcodes et simplifier la maintenance.
- Impact: meilleure reproductibilite des runs.

## Entree 2 - Correctif donnees Stellantis

- Probleme: serie quasi constante detectee dans les donnees brutes.
- Cause probable: ticker invalide/non representatif.
- Action: ajout fallback de tickers et verification de variabilite.
- Impact: fiabilisation de la collecte.

## Entree 3 - Refonte preprocessing

- Decision: normaliser le pipeline de pretraitement.
- Actions:
  - extraction prix standardisee,
  - calcul log-retours,
  - alignement sur dates communes,
  - gestion optionnelle de `TariffEvent`.
- Impact: dataset unique coherent pour l'apprentissage.

## Entree 4 - Construction du prototype fonctionnel

- Actions:
  - implementation baseline LSTM,
  - implementation enriched LSTM,
  - sauvegarde modeles et predictions.
- Impact: comparaison objective possible.

## Entree 5 - Evaluation standardisee

- Action: creation `src/evaluation/metrics.py`.
- Sorties:
  - metriques JSON,
  - comparaison CSV,
  - predictions fusionnees.
- Impact: support direct pour analyse et soutenance.

## Entree 6 - Documentation jury

- Action: creation des documents roadmap, architecture, tests, valorisation.
- Impact: couverture des attendus Notion en livrables explicites.

## Points de vigilance ouverts

1. Ajouter une source evenementielle tariff fiable et tracee.
2. Elargir les tests de robustesse temporelle.
3. Consolider le narratif metier pour la soutenance finale.
