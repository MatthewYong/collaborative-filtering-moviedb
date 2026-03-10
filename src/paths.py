from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-100k"

RATINGS_PATH = DATA_DIR / "u.data"
MOVIES_PATH = DATA_DIR / "u.item"
