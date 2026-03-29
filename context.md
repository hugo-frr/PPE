# PPE — Contexte du projet (handoff Claude Code)

## Objectif académique
Analyser l'impact des chocs tarifaires américains (2018–2025) sur les marchés actions
français, et comparer modèles statistiques vs deep learning.

**Question de recherche** : How do U.S. tariff announcements affect French equity prices,
and which models best capture and predict these effects?

---

## État d'avancement

### ✅ Fait
- Structure du projet en place (voir arborescence ci-dessous)
- `src/models/arima_model.py` — pipeline ARIMA/ARIMAX complet avec walk-forward optimisé


### 🔲 À faire (priorité ordre)
- `src/models/dl_models.py` — RNN/LSTM/GRU PyTorch avec early stopping, gradient clipping
- `src/evaluation/evaluation.py` — comparaison inter-modèles + scénarios 2025-2026
- `data/events/tariff_events.csv` — 25 événements tarifaires 2018-2025 documentés
- `data/raw/` — Airbus.csv, LVMH.csv, Stellantis.csv, CAC40.csv
- `data/processed/` — X_train.npy, X_test.npy, y_train.npy, y_test.npy
- Baseline LSTM tourne : RMSE=4.7254 | MAE=3.6847 | MAPE=0.0231
- `notebooks/03_baseline_LSTM.ipynb` — fonctionnel
1. `notebooks/04_deep_learning.ipynb` — entraîner RNN, LSTM, GRU avec les nouveaux modules
2. `notebooks/05_evaluation.ipynb` — comparaison finale + graphiques publication-ready
3. Intégrer les tariff features dans le preprocessing (retravailler les .npy)
4. Analyse de scénarios 2025-2026 (4 scénarios définis dans evaluation.py)
5. `src/models/garch_model.py` — modélisation de la volatilité (optionnel mais +++)

---

## Architecture des fichiers

```
PPE/
├── data/
│   ├── raw/          Airbus.csv, LVMH.csv, Stellantis.csv, CAC40.csv
│   ├── processed/    X_train.npy, X_test.npy, y_train.npy, y_test.npy,
│   │                 market_data_processed.csv
│   ├── events/       tariff_events.csv  ← 25 événements 2018-2025
│   └── reports/      baseline_predictions.csv, model_comparison.csv,
│                     enriched_predictions.csv, predictions_merged.csv
├── docs/             roadmap, architecture, journal d'évolution...
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_LSTM.ipynb   ← tourne, RMSE=4.7254
│   └── 05_evaluation.ipynb
├── src/
│   ├── data/
│   ├── evaluation/   evaluation.py  ← comparaison + scénarios
│   ├── models/
│   │   ├── arima_model.py    ← ARIMA/ARIMAX walk-forward O(n)
│   │   └── dl_models.py      ← RNN/LSTM/GRU PyTorch modulaire
│   ├── __init__.py
│   ├── arima.py              ← ancienne version (à remplacer par arima_model.py)
│   ├── config.py
│   ├── data_utils.py
│   └── main.py
└── tests/
```

---

## Paramètres clés (config)

```python
ASSETS        = ["Airbus", "LVMH", "Stellantis"]
INDEX         = "CAC40"
TRAIN_START   = "2018-01-02"
TRAIN_END     = "2024-12-31"
TEST_START    = "2025-01-01"
TEST_END      = "2025-11-18"
SEQ_LEN       = 60      # lookback window (jours ouvrés)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
EPOCHS        = 50
BATCH_SIZE    = 32
SEED          = 42
```

---

## Décisions techniques importantes (à respecter)

1. **Log-rendements** comme target (pas les prix) — stationnarité garantie
2. **Walk-forward expanding window** pour ARIMA — cohérent avec la littérature
3. **Pas de data leakage** — StandardScaler fit uniquement sur le train
4. **Gradient clipping** max_norm=1.0 activé pour les modèles DL
5. **Early stopping** sur val_loss (patience=10) avec restauration des meilleurs poids
6. **Métriques standardisées** : RMSE, MAE, Directional_Accuracy, Event_Window_Accuracy
   — Event_Window_Accuracy = DA spécifique aux ±5 jours autour des chocs tarifaires
   — C'est la contribution académique originale du PPE

---

## Format des données attendu

### tariff_events.csv
```
date, event_label, tariff_rate, affected_sectors, notes
2018-03-01, Steel/Aluminum 25%, 0.25, Industrials;Automotive;Aerospace, ...
```

### Fichiers CSV des actions
```
date, open, high, low, close, volume
2018-01-02, ..., ..., ..., 85.42, ...
```

### Features engineerées (dans dl_models.py)
- `log_return` — target
- `return_lag_{1,2,3,5}` — composantes AR
- `vol_5d`, `vol_20d` — volatilité réalisée
- `tariff_dummy` — 1 le jour du choc
- `tariff_intensity` — valeur du tarif (0.0 à 0.45)
- `tariff_lag_{1,5,10,30}` — transmission décalée
- `tariff_rolling_30` — régime tarifaire

---

## Prochaine tâche suggérée

Créer `notebooks/04_deep_learning.ipynb` qui :
1. Importe `from src.models.dl_models import run_multi_asset_dl`
2. Lance RNN, LSTM, GRU sur Airbus, LVMH, Stellantis
3. Affiche les courbes d'entraînement et les métriques comparatives
4. Exporte les résultats dans `data/reports/dl_comparison.csv`

Commande de lancement rapide :
```python
from src.models.dl_models import run_multi_asset_dl

results = run_multi_asset_dl(
    assets={
        "Airbus":     "data/raw/Airbus.csv",
        "LVMH":       "data/raw/LVMH.csv",
        "Stellantis": "data/raw/Stellantis.csv",
    },
    model_types=["RNN", "LSTM", "GRU"],
    events_filepath="data/events/tariff_events.csv",
    save_dir="data/reports",
)
```

---

## Références bibliographiques (pour le rapport)
- Gu, Kelly & Xiu (2020) — Empirical Asset Pricing via Machine Learning, RFS
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory, Neural Computation
- Cho et al. (2014) — Learning Phrase Representations using GRU, EMNLP
- Fama (1970) — Efficient Capital Markets, Journal of Finance
- Box & Jenkins (1976) — Time Series Analysis, Holden-Day
