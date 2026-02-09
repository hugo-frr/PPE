"""
Télécharge les données de marché et sauvegarde un CSV par actif dans data/raw/.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf

from config import END_DATE, RAW_DIR, START_DATE, TICKER_CANDIDATES


def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" in df.columns:
        series = df["Adj Close"]
    elif "Close" in df.columns:
        series = df["Close"]
    elif "Price" in df.columns:
        series = df["Price"]
    else:
        raise ValueError("Aucune colonne prix trouvée (Adj Close/Close/Price).")
    return pd.to_numeric(series, errors="coerce").dropna()


def _download_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _is_usable(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    try:
        close = _extract_close_series(df)
    except ValueError:
        return False
    return len(close) >= 120 and close.nunique() > 10


def download_market_data(
    ticker_candidates: dict[str, Iterable[str]] | None = None,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> dict[str, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    candidates = ticker_candidates or TICKER_CANDIDATES
    selected_symbols: dict[str, str] = {}

    for asset_name, symbols in candidates.items():
        print(f"[Téléchargement] {asset_name} ...")
        last_error = None

        for symbol in symbols:
            try:
                df = _download_symbol(symbol, start_date, end_date)
                if not _is_usable(df):
                    print(f"  - {symbol}: données insuffisantes.")
                    continue

                df = df.reset_index()
                output_path = RAW_DIR / f"{asset_name}.csv"
                df.to_csv(output_path, index=False)
                selected_symbols[asset_name] = symbol

                print(f"  - OK {symbol} -> {output_path} ({len(df)} lignes)")
                break
            except Exception as exc:  # pragma: no cover (dépend d'API externe)
                last_error = exc
                print(f"  - {symbol}: échec ({exc})")

        if asset_name not in selected_symbols:
            raise RuntimeError(
                f"Aucun ticker valide trouvé pour {asset_name}. "
                f"Dernière erreur: {last_error}"
            )

    return selected_symbols


def main() -> None:
    print("=== Téléchargement des données de marché ===")
    selected = download_market_data()
    print("=== Terminé ===")
    for asset_name, symbol in selected.items():
        print(f"  {asset_name}: {symbol}")


if __name__ == "__main__":
    main()
