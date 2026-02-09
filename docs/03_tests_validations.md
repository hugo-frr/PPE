# Tests et Validations

## 1. Strategie de test

Trois niveaux sont utilises:

1. Tests unitaires de pipeline data (formats, colonnes, sequences).
2. Test d'integration (execution `src/main.py`).
3. Validation experimentale (comparaison baseline vs enriched).

## 2. Protocole de validation experimentale

- Jeu de donnees: `data/processed/market_data_processed.csv`
- Cible: `Airbus_price`
- Split: temporel (80% train, 20% test)
- Metriques: RMSE, MAE, MAPE
- Modeles compares:
  - `baseline_lstm`
  - `enriched_lstm`

## 3. Resultats observes (run local)

Source: `data/reports/model_comparison.csv`

- Baseline LSTM:
  - RMSE: 4.7254
  - MAE: 3.6847
  - MAPE: 0.0231
- Enriched LSTM:
  - RMSE: 4.8835
  - MAE: 3.7304
  - MAPE: 0.0226

Interpretation:
- La baseline est legerement meilleure sur RMSE/MAE.
- Le modele enrichi est legerement meilleur sur MAPE.
- La feature `TariffEvent` doit etre enrichie pour augmenter sa valeur predictive.

## 4. Critere d'acceptation de prototype

- Le pipeline s'execute de bout en bout sans erreur bloquante.
- Les artefacts modeles et rapports sont generes.
- Les performances sont mesurables et comparables.
- Les limites sont explicites et tracees.

## 5. Limites actuelles

- Dataset d'evenements tarifaires encore simplifie (`TariffEvent` binaire).
- Un seul horizon de prediction evalue.
- Peu de baselines statistiques de reference.

## 6. Plan d'amelioration

1. Introduire une validation walk-forward.
2. Ajouter baseline naive et moyenne mobile.
3. Mesurer robustesse par regime de volatilite.
4. Ajouter tests de sensibilite sur `SEQ_LEN` et fenetre temporelle.
