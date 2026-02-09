# Document d'Architecture Technique

## 1. Objectif systeme

Construire un pipeline reproducible qui:

1. telecharge les donnees financieres,
2. prepare un dataset aligne,
3. entraine deux modeles LSTM,
4. compare leurs performances,
5. exporte des artefacts pour analyse et soutenance.

## 2. Architecture logique

- Couche configuration:
  - `src/config.py`
- Couche donnees:
  - `src/data/download_data.py`
  - `src/data/process_data.py`
  - `src/data_utils.py`
- Couche modeles:
  - `src/models/lstm_baseline.py`
  - `src/models/lstm_enriched.py`
- Couche evaluation:
  - `src/evaluation/metrics.py`
- Orchestration:
  - `src/main.py`

## 3. Flux de donnees

1. Yahoo Finance -> `data/raw/*.csv`
2. Nettoyage + log-retours + `TariffEvent` -> `data/processed/market_data_processed.csv`
3. Entrainement baseline/enriched -> `data/models/*.keras`
4. Export metriques/predictions -> `data/reports/*.json` et `*.csv`

## 4. Choix techniques

- Python 3.13 + TensorFlow/Keras.
- Split temporel (train passe, test futur).
- MinMaxScaler fite uniquement sur train pour limiter la fuite de donnees.
- Early stopping pour limiter le surapprentissage.
- Sauvegarde des artefacts pour reproductibilite.

## 5. Interfaces d'execution

- Pipeline complet:
  - `python3 src/main.py`
- Sans telechargement:
  - `python3 src/main.py --skip-download`
- Reentrainement force:
  - `python3 src/main.py --skip-download --force-retrain`

## 6. Qualite et robustesse

- Fallback ticker pour Stellantis pour eviter les series plates.
- Validation des colonnes minimales avant traitement.
- Contrats simples via tests automatiques (voir `tests/`).

## 7. Positionnement TRL

Niveau cible: TRL5 (prototype fonctionnel dans un environnement representatif).

Justification:
- prototype operationnel de bout en bout,
- donnees reelles de marche,
- metriques quantitatives exportees,
- documentation technique et guide de prise en main.

Limites a lever vers TRL6+:
- industrialisation du monitoring,
- gestion des incidents de donnees,
- benchmark elargi sur plusieurs periodes de stress de marche.
