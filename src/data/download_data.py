"""
Script : download_data.py
Objectif : Télécharger automatiquement les données historiques
           de Airbus, LVMH, Stellantis et du CAC40.

Actions françaises :
 - Airbus      : AIR.PA
 - LVMH        : MC.PA
 - Stellantis  : STLAM.MI  (cotée à Milan) OU STLA.PA selon Yahoo Finance

Indice :
 - CAC40       : ^FCHI

Format de sortie : CSV dans data/raw/
"""

import yfinance as yf
import os

# -----------------------------
# 1. Configuration
# -----------------------------

TICKERS = {
    
    "Stellantis": "STLAP.PA",  # Stellantis Paris (CAC40)
}

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = "data/raw/"


# -----------------------------
# 2. Création du dossier si besoin
# -----------------------------

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"[+] Dossier créé : {OUTPUT_DIR}")


# -----------------------------
# 3. Téléchargement des données
# -----------------------------

def download_and_save(ticker_name, ticker_symbol):
    print(f"[Téléchargement] {ticker_name} ({ticker_symbol}) ...")

    data = yf.download(
        ticker_symbol,
        start=START_DATE,
        end=END_DATE,
        progress=True
    )

    file_path = os.path.join(OUTPUT_DIR, f"{ticker_name}.csv")
    data.to_csv(file_path)

    print(f"[OK] Données sauvegardées : {file_path}\n")


def main():
    print("\n=== Téléchargement des données financières ===\n")

    for name, symbol in TICKERS.items():
        download_and_save(name, symbol)

    print("\n=== Terminé ! ===\n")


if __name__ == "__main__":
    main()