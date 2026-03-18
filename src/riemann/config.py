from pathlib import Path

# Precision -- user decision: "this problem will be cracked in under 50 digits"
DEFAULT_DPS = 50

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "zeros.db"
CACHE_DIR = DATA_DIR / "cache"
COMPUTED_DIR = DATA_DIR / "computed"
ODLYZKO_DIR = DATA_DIR / "odlyzko"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure data directories exist on import
for d in [DATA_DIR, CACHE_DIR, COMPUTED_DIR, ODLYZKO_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
