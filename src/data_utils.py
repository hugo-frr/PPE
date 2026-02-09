from __future__ import annotations

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, SEQ_LEN, TEST_RATIO
from data.process_data import process_and_save as _process_and_save


def processed_data_path() -> str:
    return str(PROCESSED_DIR / "market_data_processed.csv")


def preprocess_and_save(force: bool = False) -> pd.DataFrame:
    output_path = PROCESSED_DIR / "market_data_processed.csv"
    if force or not output_path.exists():
        return _process_and_save()
    return load_processed_dataframe()


def load_processed_dataframe() -> pd.DataFrame:
    output_path = PROCESSED_DIR / "market_data_processed.csv"
    if not output_path.exists():
        return _process_and_save()

    df = pd.read_csv(output_path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def train_cutoff_index(length: int, test_ratio: float = TEST_RATIO) -> int:
    if length < 2:
        raise ValueError("Le dataset est trop petit pour un split train/test.")
    split_idx = int(length * (1 - test_ratio))
    return max(1, min(split_idx, length - 1))


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) != len(target):
        raise ValueError("features et target doivent avoir la même longueur.")
    if len(features) <= seq_len:
        raise ValueError("Pas assez de lignes pour créer des séquences.")

    X, y, idx = [], [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len : i])
        y.append(target[i])
        idx.append(i)

    return np.array(X), np.array(y), np.array(idx)
