import os
from pathlib import Path

# repo root: src/energy_prediction/config.py -> parents[2]; override with ENERGY_ROOT
ROOT = Path(os.getenv("ENERGY_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
ARTIFACTS_DIR = ROOT / "artifacts"
