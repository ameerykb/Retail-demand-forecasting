from pathlib import Path
import pandas as pd
from src.data.rossmann import load_rossmann, standardize_rossmann
from src.data.synthetic import make_rossmann_like
from src.data.load import to_wide
from src.features.build import make_features
from src.training.backtest import summarize
from src.training.backtest_gbm import rolling_backtest_gbm

HORIZON = 42

if Path("data/train.csv").exists():
    df, source = load_rossmann("data/train.csv", "data/store.csv"), "real Rossmann data"
else:
    df, source = standardize_rossmann(*make_rossmann_like(n_stores=20, n_days=700)), "SYNTHETIC data"
print(f"Using {source}: {len(df):,} rows, {df['store'].nunique()} stores")

wide = to_wide(df)
df, feature_cols = make_features(df, horizon=HORIZON)
print("Features:", feature_cols)

results, model = rolling_backtest_gbm(df, wide, feature_cols, horizon=HORIZON, n_folds=4)
print("\nPer-fold results (open days only):\n", results.to_string(index=False))
print("\nAverage across folds (lower is better):\n", summarize(results))

imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance (last fold):\n", imp.to_string())
