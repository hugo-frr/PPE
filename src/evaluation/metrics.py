from __future__ import annotations

import pandas as pd

from config import REPORTS_DIR
from models.lstm_baseline import train_baseline
from models.lstm_enriched import train_enriched


def _comparison_dataframe(
    baseline_metrics: dict[str, float],
    enriched_metrics: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "baseline_lstm", **baseline_metrics},
            {"model": "enriched_lstm", **enriched_metrics},
        ]
    )


def _save_merged_predictions(
    baseline_predictions: pd.DataFrame,
    enriched_predictions: pd.DataFrame,
) -> pd.DataFrame:
    baseline_df = baseline_predictions.rename(columns={"y_pred": "y_pred_baseline"}).copy()
    enriched_df = enriched_predictions.rename(columns={"y_pred": "y_pred_enriched"}).copy()

    baseline_df["Date"] = pd.to_datetime(baseline_df["Date"])
    enriched_df["Date"] = pd.to_datetime(enriched_df["Date"])

    plot_df = (
        baseline_df[["Date", "y_true", "y_pred_baseline"]]
        .merge(enriched_df[["Date", "y_pred_enriched"]], on="Date", how="inner")
        .sort_values("Date")
    )

    return plot_df


def evaluate_models(force_retrain: bool = False) -> tuple[float, float, float, float]:
    baseline = train_baseline(force_retrain=force_retrain, return_artifacts=True)
    enriched = train_enriched(force_retrain=force_retrain, return_artifacts=True)

    comparison = _comparison_dataframe(baseline["metrics"], enriched["metrics"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    merged_predictions = _save_merged_predictions(
        baseline["predictions"], enriched["predictions"]
    )
    merged_predictions.to_csv(REPORTS_DIR / "predictions_merged.csv", index=False)

    print(f"[OK] Comparaison sauvegardée: {comparison_path}")
    return (
        baseline["metrics"]["rmse"],
        enriched["metrics"]["rmse"],
        baseline["metrics"]["mae"],
        enriched["metrics"]["mae"],
    )
