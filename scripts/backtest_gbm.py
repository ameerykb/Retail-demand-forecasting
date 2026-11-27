"""LightGBM vs baselines in the rolling backtest, plus feature importance.

    python -m scripts.backtest_gbm
"""
import pandas as pd
from src import config
from src.data.rossmann import load_data, to_wide
from src.features.build import make_features
from src.models.gbm import load_params
from src.training.backtest import rolling_backtest_gbm, summarize


def main():
    df, source = load_data()
    print(f"Using {source}: {len(df):,} rows, {df['store'].nunique()} stores")
    wide = to_wide(df)
    df, cols = make_features(df, horizon=config.HORIZON)
    params = load_params(config.BEST_PARAMS)
    print("LightGBM params:", "tuned" if config.BEST_PARAMS.exists() else "defaults")

    results, model = rolling_backtest_gbm(df, wide, cols, horizon=config.HORIZON,
                                          n_folds=config.N_FOLDS, params=params)
    print("\nPer-fold results (open days only):\n", results.to_string(index=False))
    print("\nAverage across folds (lower is better):\n", summarize(results))
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nFeature importance, split count (last fold):\n", imp.to_string())


if __name__ == "__main__":
    main()
