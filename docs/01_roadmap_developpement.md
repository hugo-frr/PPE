# Roadmap de Developpement PPE

## Contexte

Projet technique de prediction de prix Airbus a partir de donnees de marche
(Airbus, LVMH, Stellantis, CAC40) et variable evenementielle `TariffEvent`.

Objectif de la phase Developpement: obtenir un prototype fonctionnel,
testable, documente et presentable en soutenance finale.

## Methode de pilotage

- Cadence: iterations hebdomadaires (sprints courts).
- Outil de suivi: backlog priorise + point equipe hebdomadaire.
- Regle: chaque sprint produit un artefact verifiable (code, test, rapport).

## Sprint 1 - Stabilisation technique

- Mettre en place une configuration centrale.
- Corriger la collecte des donnees (fallback ticker Stellantis).
- Produire un dataset nettoye unique.
- Definition of Done:
  - `data/processed/market_data_processed.csv` genere.
  - Colonnes attendues presentes.
  - Script de pipeline executable.

## Sprint 2 - Prototype fonctionnel TRL5

- Implementer 2 modeles:
  - baseline LSTM (univarie),
  - enriched LSTM (multivarie).
- Exporter metriques et predictions.
- Definition of Done:
  - Modeles sauvegardes (`data/models/`),
  - metriques comparatives (`data/reports/model_comparison.csv`),
  - pipeline end-to-end lance sans erreur.

## Sprint 3 - Tests et validations

- Ecrire des tests automatiques de non-regression.
- Definir un protocole de validation experimental.
- Produire une table de resultats interpretable.
- Definition of Done:
  - dossier `tests/` execute,
  - document de validation rempli,
  - limites du modele explicitees.

## Sprint 4 - Valorisation et soutenance

- Produire document d'architecture.
- Produire guide de prise en main.
- Rediger storyboard/script video 60s.
- Construire le plan de soutenance finale.
- Definition of Done:
  - dossier `docs/` complet,
  - elements de valorisation prets,
  - checklist rendus completee.

## Backlog priorise (reste a faire)

1. Ajouter un vrai fichier `data/external/tariff_events.csv` source et versionne.
2. Ajouter des baselines statistiques (naive, moyenne mobile) pour comparaison.
3. Ajouter validation temporelle glissante (walk-forward).
4. Ajouter visualisations de performance (erreur par periode, drift).
5. Ajouter script de generation automatique du dossier final.

## Risques et mitigation

- Donnees externes instables (Yahoo Finance):
  - mitigation: fallback ticker + verification de variabilite des series.
- Surapprentissage:
  - mitigation: split temporel strict + early stopping + comparaison baseline.
- Valeur faible de la feature evenementielle:
  - mitigation: enrichir `TariffEvent` (intensite, duree, source officielle).
