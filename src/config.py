# src/config.py

from pathlib import Path

# Racine du projet (à ADAPTER si ton chemin est différent)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossiers de données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Création des dossiers si besoin
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Tickers étudiés
TICKERS = ["AIR.PA", "MC.PA", "STLAM.MI"]  # Airbus, LVMH, Stellantis

# Fenêtre de temps
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# Longueur des séquences pour LSTM
SEQ_LEN = 60