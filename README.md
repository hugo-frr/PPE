# PPE - Prédiction de prix Airbus avec LSTM

Ce dépôt implémente une solution de bout en bout pour:
- télécharger des données de marché (Airbus, LVMH, Stellantis, CAC40),
- construire un dataset aligné avec indicateur `TariffEvent`,
- entraîner un modèle baseline LSTM,
- entraîner un modèle enrichi LSTM (variables explicatives additionnelles),
- comparer les performances (RMSE, MAE, MAPE).

## Structure

```text
src/
  config.py
  main.py
  data/
    download_data.py
    process_data.py
  models/
    lstm_baseline.py
    lstm_enriched.py
  evaluation/
    metrics.py
data/
  raw/
  processed/
  models/
  reports/
  figures/
notebooks/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Exécution complète

Depuis la racine du projet:

```bash
python3 src/main.py
```

Options utiles:

```bash
python3 src/main.py --skip-download
python3 src/main.py --skip-download --force-retrain
```

## Tests

Executer les tests automatiques:

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

## Artefacts générés

- Dataset final: `data/processed/market_data_processed.csv`
- Modèles: `data/models/baseline_lstm.keras`, `data/models/enriched_lstm.keras`
- Métriques individuelles: `data/reports/baseline_metrics.json`, `data/reports/enriched_metrics.json`
- Comparaison: `data/reports/model_comparison.csv`
- Prédictions: `data/reports/baseline_predictions.csv`, `data/reports/enriched_predictions.csv`
- Fusion des prédictions: `data/reports/predictions_merged.csv`

## Dossier jury

- Roadmap: `docs/01_roadmap_developpement.md`
- Architecture: `docs/02_architecture_technique.md`
- Tests/validation: `docs/03_tests_validations.md`
- Journal d'evolution: `docs/04_journal_evolution.md`
- Valorisation + video 60s: `docs/05_valorisation_video_60s.md`
- Checklist finale: `docs/06_checklist_rendus_finaux.md`

## Notebooks

Les notebooks existants peuvent être utilisés pour la narration du projet:
- exploration des données,
- préprocessing,
- baseline,
- modèle enrichi,
- évaluation.

Les fonctions exposées côté code:
- `data_utils.preprocess_and_save()`
- `models.lstm_baseline.train_baseline()`
- `models.lstm_enriched.train_enriched()`
- `evaluation.metrics.evaluate_models()`
