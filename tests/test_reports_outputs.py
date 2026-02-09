import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


class TestReportsOutputs(unittest.TestCase):
    def test_comparison_report_exists_and_has_expected_columns(self):
        csv_path = REPORTS_DIR / "model_comparison.csv"
        self.assertTrue(csv_path.exists(), f"Fichier manquant: {csv_path}")

        df = pd.read_csv(csv_path)
        expected = {"model", "rmse", "mae", "mape"}
        self.assertTrue(expected.issubset(set(df.columns)))
        self.assertGreaterEqual(len(df), 2)

    def test_predictions_merged_exists(self):
        csv_path = REPORTS_DIR / "predictions_merged.csv"
        self.assertTrue(csv_path.exists(), f"Fichier manquant: {csv_path}")
        df = pd.read_csv(csv_path)
        expected = {"Date", "y_true", "y_pred_baseline", "y_pred_enriched"}
        self.assertTrue(expected.issubset(set(df.columns)))
        self.assertGreater(len(df), 50)


if __name__ == "__main__":
    unittest.main()
