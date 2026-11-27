"""Train the final LightGBM on ALL available data and save the serving bundle.

    python train_final.py                  # real data if data/train.csv exists, else synthetic
    python train_final.py --synthetic      # force synthetic data (used in CI)
    python train_final.py --backtest-wape 0.0912   # record your backtest score in /model-info
"""
import argparse
from pathlib import Path
from src.api.bundle import save_bundle
from src.data.rossmann import load_rossmann, standardize_rossmann
from src.data.synthetic import make_rossmann_like
from src.features.build import make_features
from src.models.gbm import train_lgbm

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--backtest-wape", type=float, default=None)
ap.add_argument("--out", default="models/forecaster")
args = ap.parse_args()
HORIZON = 42

if Path("data/train.csv").exists() and not args.synthetic:
    raw, source = load_rossmann("data/train.csv", "data/store.csv"), "Rossmann Store Sales (Kaggle)"
else:
    raw, source = standardize_rossmann(*make_rossmann_like(n_stores=20, n_days=700)), "synthetic"
print(f"Training on {source}: {len(raw):,} rows")

df, cols = make_features(raw, horizon=HORIZON)
model = train_lgbm(df, cols)
out = save_bundle(model, cols, raw, HORIZON, args.out, trained_on=source,
                  backtest_wape=args.backtest_wape)
print(f"Saved model bundle to {out}")
