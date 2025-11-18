"""
Script : process_data.py
Objectif :
    - Charger les données brutes (Airbus, LVMH, Stellantis, CAC40) depuis data/raw/
    - Utiliser les fichiers CSV générés par yfinance :
        Date,Open,High,Low,Close,Adj Close,Volume
    - Calculer les log-rendements
    - Aligner toutes les séries sur les mêmes dates
    - Ajouter une colonne TariffEvent (0 par défaut)
    - Sauvegarder un fichier unique dans data/processed/market_data_processed.csv
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

ASSETS = ["Airbus", "LVMH", "Stellantis"]
INDEX_NAME = "CAC40"

# Si un jour tu crées un fichier d’événements : data/external/tariff_events.csv
TARIFF_EVENTS_FILE = "data/external/tariff_events.csv"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_asset(name: str) -> pd.DataFrame:
    """
    Charge un CSV yfinance standard de la forme :

    Date,Open,High,Low,Close,Adj Close,Volume

    et retourne un DataFrame indexé par Date avec :
        - {name}_price
        - {name}_ret
    """
    file_path = os.path.join(RAW_DIR, f"{name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"[Chargement] {file_path}")
    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        raise ValueError(f"Colonne 'Date' manquante dans {file_path}")

    # Conversion en datetime + tri
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Choix de la colonne de prix : Adj Close si possible, sinon Close
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"Aucune colonne de prix ('Adj Close' ou 'Close') trouvée dans {file_path}")

    # Conversion en float
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    # Renommage
    out = df[[price_col]].rename(columns={price_col: f"{name}_price"})

    # Log-rendements
    out[f"{name}_ret"] = np.log(out[f"{name}_price"] / out[f"{name}_price"].shift(1))

    return out


def add_tariff_event_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne TariffEvent :
      - Si data/external/tariff_events.csv existe :
            fusionne les événements
      - Sinon :
            met 0 partout
    Format attendu pour tariff_events.csv :
        Date,TariffEvent
        2025-03-01,1
        2025-06-15,1
    """
    if os.path.exists(TARIFF_EVENTS_FILE):
        print(f"[Info] Fichier d'événements trouvé : {TARIFF_EVENTS_FILE}")
        events = pd.read_csv(TARIFF_EVENTS_FILE)
        if "Date" not in events.columns or "TariffEvent" not in events.columns:
            raise ValueError("Le fichier tariff_events.csv doit contenir les colonnes 'Date' et 'TariffEvent'.")

        events["Date"] = pd.to_datetime(events["Date"], errors="coerce")
        events = events.dropna(subset=["Date"]).set_index("Date").sort_index()

        df["TariffEvent"] = 0
        df.update(events[["TariffEvent"]])
    else:
        print("[Info] Aucun fichier d'événements trouvé. TariffEvent = 0.")
        df["TariffEvent"] = 0

    return df


def main():
    print("\n=== Prétraitement des données de marché ===\n")

    ensure_dir(PROCESSED_DIR)

    # 1. Charger les actifs individuels
    all_dfs = []
    for asset in ASSETS:
        df_asset = load_asset(asset)
        all_dfs.append(df_asset)

    # 2. Charger le CAC40
    df_index = load_asset(INDEX_NAME)
    all_dfs.append(df_index)

    # 3. Fusion sur les dates communes
    print("[Fusion] Alignement des séries temporelles sur les dates communes...")
    df_merged = all_dfs[0]
    for df in all_dfs[1:]:
        df_merged = df_merged.join(df, how="inner")

    # 4. Suppression des premières lignes NaN dues aux shifts
    df_merged = df_merged.dropna()

    # 5. Ajout de TariffEvent
    df_merged = add_tariff_event_column(df_merged)

    # 6. Sauvegarde
    output_file = os.path.join(PROCESSED_DIR, "market_data_processed.csv")
    df_merged.to_csv(output_file)

    print(f"\n[OK] Fichier final sauvegardé : {output_file}")
    print("Colonnes :", list(df_merged.columns))
    print(f"Nombre de lignes : {len(df_merged)}")
    print("\n=== Terminé ===\n")


if __name__ == "__main__":
    main()