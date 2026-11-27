"""Project-wide settings in one place (instead of repeating them in every script)."""
from pathlib import Path

HORIZON = 42                     # days ahead we forecast; every sales lag is >= this
N_FOLDS = 4                      # rolling-backtest folds
QUANTILES = (0.1, 0.9)           # prediction interval: 80% range (p10 to p90)

DATA_DIR = Path("data")
TRAIN_CSV = DATA_DIR / "train.csv"
STORE_CSV = DATA_DIR / "store.csv"
MODEL_DIR = Path("models/forecaster")
BEST_PARAMS = Path("configs/lgbm_best_params.json")   # written by scripts/tune_lgbm.py
FIGURES = Path("reports/figures")
