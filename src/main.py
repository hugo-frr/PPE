from __future__ import annotations

import argparse

from data.download_data import download_market_data
from data.process_data import process_and_save
from evaluation.metrics import evaluate_models


def run_pipeline(skip_download: bool = False, force_retrain: bool = False) -> None:
    if not skip_download:
        print("=== Étape 1/3 : téléchargement ===")
        download_market_data()

    print("=== Étape 2/3 : prétraitement ===")
    process_and_save()

    print("=== Étape 3/3 : entraînement + évaluation ===")
    rmse_base, rmse_enriched, mae_base, mae_enriched = evaluate_models(
        force_retrain=force_retrain
    )
    print("=== Résultats ===")
    print(f"Baseline   -> RMSE={rmse_base:.4f} | MAE={mae_base:.4f}")
    print(f"Enriched   -> RMSE={rmse_enriched:.4f} | MAE={mae_enriched:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline complet PPE")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="N'effectue pas le téléchargement Yahoo Finance.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Réentraine les modèles même si des artefacts existent déjà.",
    )
    args = parser.parse_args()

    run_pipeline(skip_download=args.skip_download, force_retrain=args.force_retrain)


if __name__ == "__main__":
    main()
