"""Baselines only: the scores any real model has to beat.

    python -m scripts.baselines
"""
from src import config
from src.data.rossmann import load_data, to_wide
from src.features.build import make_features
from src.training.backtest import rolling_backtest_models, summarize


def main():
    df, source = load_data()
    print(f"Using {source}: {len(df):,} rows, {df['store'].nunique()} stores")
    wide = to_wide(df)
    df, cols = make_features(df, horizon=config.HORIZON)
    results = rolling_backtest_models(df, wide, cols, learners={},
                                      horizon=config.HORIZON, n_folds=config.N_FOLDS)
    print("\nPer-fold results (open days only):\n", results.to_string(index=False))
    print("\nAverage across folds (lower is better):\n", summarize(results))


if __name__ == "__main__":
    main()
