import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_utils import create_sequences, load_processed_dataframe, preprocess_and_save


class TestDataPipeline(unittest.TestCase):
    def test_preprocess_returns_dataframe_with_expected_columns(self):
        df = preprocess_and_save(force=False)
        expected = {
            "Airbus_price",
            "Airbus_ret",
            "LVMH_price",
            "LVMH_ret",
            "Stellantis_price",
            "Stellantis_ret",
            "CAC40_price",
            "CAC40_ret",
            "TariffEvent",
        }
        self.assertTrue(expected.issubset(set(df.columns)))
        self.assertGreater(len(df), 100)

    def test_load_processed_dataframe_has_datetime_index(self):
        df = load_processed_dataframe()
        self.assertEqual(df.index.name, "Date")
        self.assertGreater(len(df.index.unique()), 100)

    def test_create_sequences_shapes(self):
        features = np.arange(100, dtype=np.float32).reshape(-1, 1)
        target = np.arange(100, dtype=np.float32).reshape(-1, 1)
        seq_len = 10

        X, y, idx = create_sequences(features, target, seq_len=seq_len)
        self.assertEqual(X.shape[0], 90)
        self.assertEqual(X.shape[1], seq_len)
        self.assertEqual(X.shape[2], 1)
        self.assertEqual(y.shape[0], 90)
        self.assertEqual(idx[0], seq_len)
        self.assertEqual(idx[-1], 99)


if __name__ == "__main__":
    unittest.main()
