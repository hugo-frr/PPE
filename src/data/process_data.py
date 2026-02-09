"""
Prépare un dataset unique aligné sur dates communes:
- prix
- log-rendements
- indicateur TariffEvent (optionnel)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ASSET_NAMES, DATA_DIR, INDEX_NAME, PROCESSED_DIR, RAW_DIR

TARIFF_EVENTS_FILE = DATA_DIR / "external" / "tariff_events.csv"


def _extract_price_column(df: pd.DataFrame) -> str:
    for candidate in ("Adj Close", "Close", "Price"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Aucune colonne de prix trouvée (Adj Close/Close/Price).")


def load_asset(asset_name: str) -> pd.DataFrame:
    csv_path = RAW_DIR / f"{asset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError(f"Colonne 'Date' manquante dans {csv_path}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    price_col = _extract_price_column(df)
    price = pd.to_numeric(df[price_col], errors="coerce").dropna()

    out = pd.DataFrame(index=price.index)
    out[f"{asset_name}_price"] = price
    out[f"{asset_name}_ret"] = np.log(price / price.shift(1))
    return out


def add_tariff_event_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TariffEvent"] = 0

    if not TARIFF_EVENTS_FILE.exists():
        return out

    events = pd.read_csv(TARIFF_EVENTS_FILE)
    required = {"Date", "TariffEvent"}
    if not required.issubset(events.columns):
        raise ValueError(
            "Le fichier tariff_events.csv doit contenir les colonnes Date et TariffEvent."
        )

    events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
    events = events.dropna(subset=["Date"])
    events["TariffEvent"] = pd.to_numeric(events["TariffEvent"], errors="coerce").fillna(0)
    events = events.groupby("Date", as_index=True)["TariffEvent"].max()

    out = out.join(events.rename("TariffEvent"), how="left", rsuffix="_event")
    if "TariffEvent_event" in out.columns:
        out["TariffEvent"] = out["TariffEvent_event"].fillna(out["TariffEvent"])
        out = out.drop(columns=["TariffEvent_event"])
    out["TariffEvent"] = out["TariffEvent"].fillna(0).astype(int)
    return out


def build_market_dataframe() -> pd.DataFrame:
    frames = [load_asset(name) for name in ASSET_NAMES + [INDEX_NAME]]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.join(frame, how="inner")
    merged = merged.dropna()
    merged = add_tariff_event_column(merged)
    return merged


def process_and_save() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_market_dataframe()
    output_path = PROCESSED_DIR / "market_data_processed.csv"
    df.to_csv(output_path)
    print(f"[OK] Dataset sauvegardé: {output_path} ({len(df)} lignes)")
    return df


def main() -> None:
    print("=== Prétraitement des données ===")
    process_and_save()
    print("=== Terminé ===")


if __name__ == "__main__":
    main()
