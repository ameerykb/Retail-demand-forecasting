"""Check the 80% prediction interval in the rolling backtest: how often does the
real value land inside [p10, p90]? Well calibrated = close to 0.80.
Shows the raw quantile models and the conformally calibrated range the API serves.

    python -m scripts.evaluate_intervals
"""
from src import config
from src.data.rossmann import load_data, to_wide
from src.features.build import make_features
from src.models.gbm import load_params
from src.training.backtest import backtest_intervals


def main():
    df, source = load_data()
    print(f"Using {source}: {len(df):,} rows")
    wide = to_wide(df)
    df, cols = make_features(df, horizon=config.HORIZON)
    res = backtest_intervals(df, wide, cols, config.QUANTILES, config.HORIZON,
                             config.N_FOLDS, load_params(config.BEST_PARAMS))
    lo, hi = config.QUANTILES
    print(f"\nInterval p{int(lo * 100)}-p{int(hi * 100)}, target coverage {hi - lo:.0%}:")
    print(res.to_string(index=False))
    m = res.mean(numeric_only=True)
    print(f"\nAverage coverage: raw {m['raw_coverage']:.3f} -> calibrated "
          f"{m['calibrated_coverage']:.3f}   (average width {m['raw_width']:,.0f} -> "
          f"{m['calibrated_width']:,.0f})")


if __name__ == "__main__":
    main()
