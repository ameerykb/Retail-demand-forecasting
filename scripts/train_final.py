"""Train the final models on ALL available data and save the serving bundle:
the main LightGBM forecast plus two quantile models for the 80% prediction interval,
calibrated on the most recent HORIZON days (see src/models/intervals.py).

    python -m scripts.train_final                          # real data if data/train.csv exists
    python -m scripts.train_final --synthetic              # synthetic data (used in CI)
    python -m scripts.train_final --backtest-wape 0.0912   # record the score shown by /model-info
"""
import argparse
from pathlib import Path
from src import config
from src.api.bundle import save_bundle
from src.data.rossmann import load_data
from src.features.build import make_features
from src.models.gbm import load_params, train_lgbm
from src.models.intervals import calibrate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--backtest-wape", type=float, default=None)
    ap.add_argument("--out", default=str(config.MODEL_DIR))
    ap.add_argument("--no-intervals", action="store_true", help="skip the quantile models")
    args = ap.parse_args()

    raw, source = load_data(synthetic=args.synthetic)
    print(f"Training on {source}: {len(raw):,} rows")
    tuned = config.BEST_PARAMS.exists() and not args.synthetic
    params = load_params(config.BEST_PARAMS if tuned else None)
    print("LightGBM params:", "tuned (configs/lgbm_best_params.json)" if tuned else "defaults")

    df, cols = make_features(raw, horizon=config.HORIZON)
    model = train_lgbm(df, cols, params)
    quantile_models, offset = {}, 0.0
    if not args.no_intervals:
        print(f"Calibrating the interval on the last {config.HORIZON} days ...")
        offset = calibrate(df, cols, config.QUANTILES, params, calib_days=config.HORIZON)
        print(f"Conformal offset (log scale): {offset:.4f}")
        for q in config.QUANTILES:
            print(f"Training quantile model q={q} ...")
            quantile_models[q] = train_lgbm(df, cols, params, quantile=q)

    out = save_bundle(model, cols, raw, config.HORIZON, Path(args.out), trained_on=source,
                      backtest_wape=args.backtest_wape, quantile_models=quantile_models,
                      params_source="tuned" if tuned else "default", interval_offset=offset)
    print(f"Saved model bundle to {out}")


if __name__ == "__main__":
    main()
