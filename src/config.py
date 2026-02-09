from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
FIGURES_DIR = DATA_DIR / "figures"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TICKER_CANDIDATES = {
    "Airbus": ["AIR.PA"],
    "LVMH": ["MC.PA"],
    "Stellantis": ["STLAM.MI", "STLAP.PA", "STLA"],
    "CAC40": ["^FCHI"],
}

ASSET_NAMES = ["Airbus", "LVMH", "Stellantis"]
INDEX_NAME = "CAC40"

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

SEQ_LEN = 60
TEST_RATIO = 0.2
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_SEED = 42
