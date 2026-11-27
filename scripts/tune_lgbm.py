"""Tune LightGBM with Optuna, without peeking at the reported backtest.

The reported backtest uses the last N_FOLDS x HORIZON days. Tuning only sees the
data BEFORE that period and runs its own small rolling backtest there, so the
reported score stays an honest out-of-sample number.

    python -m scripts.tune_lgbm --trials 25            # writes configs/lgbm_best_params.json
    python -m scripts.tune_lgbm --trials 25 --stores 300   # faster: tune on a random store sample
"""
import argparse
import datetime as dt
import json
import numpy as np
import optuna
from src import config
from src.data.rossmann import load_data, to_wide
from src.features.build import make_features
from src.models.gbm import DEFAULT_PARAMS, predict_lgbm, train_lgbm
from src.training.backtest import rolling_backtest_models


def search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tuning_data(df, holdout_days: int):
    """Everything before the reported backtest period."""
    cutoff = df["date"].max() - np.timedelta64(holdout_days - 1, "D")
    return df[df["date"] < cutoff]


def run(trials: int, folds: int, stores: int | None, seed: int = 42, synthetic: bool = False):
    raw, source = load_data(synthetic=synthetic)
    if stores:
        rng = np.random.default_rng(seed)
        keep = rng.choice(raw["store"].unique(), size=min(stores, raw["store"].nunique()),
                          replace=False)
        raw = raw[raw["store"].isin(keep)]
    raw = tuning_data(raw, config.N_FOLDS * config.HORIZON)
    print(f"Tuning on {source}: {len(raw):,} rows before the reported backtest period")
    wide = to_wide(raw)
    df, cols = make_features(raw, horizon=config.HORIZON)

    def objective(trial):
        params = {**DEFAULT_PARAMS, **search_space(trial)}
        learner = {"lgb": lambda tr, te, f: predict_lgbm(train_lgbm(tr, f, params), te, f)}
        res = rolling_backtest_models(df, wide, cols, learner, horizon=config.HORIZON,
                                      n_folds=folds, baselines=False)
        return float(res["WAPE"].mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    default_wape = objective(optuna.trial.FixedTrial({
        "learning_rate": DEFAULT_PARAMS["learning_rate"], "n_estimators": DEFAULT_PARAMS["n_estimators"],
        "num_leaves": DEFAULT_PARAMS["num_leaves"], "min_child_samples": 20,
        "subsample": DEFAULT_PARAMS["subsample"], "colsample_bytree": DEFAULT_PARAMS["colsample_bytree"],
        "reg_lambda": 1e-3}))
    print(f"Default params, tuning-period WAPE: {default_wape:.4f}")
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    print(f"Best tuning-period WAPE: {study.best_value:.4f}  (default {default_wape:.4f})")
    return study, default_wape, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--folds", type=int, default=2, help="folds inside the tuning period")
    ap.add_argument("--stores", type=int, default=None, help="tune on a random sample of stores")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    study, default_wape, source = run(args.trials, args.folds, args.stores,
                                      synthetic=args.synthetic)
    config.BEST_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    config.BEST_PARAMS.write_text(json.dumps({
        "params": study.best_params,
        "tuning_wape": round(study.best_value, 4),
        "default_wape": round(default_wape, 4),
        "trials": args.trials, "folds": args.folds, "stores": args.stores or "all",
        "data": source, "created": dt.date.today().isoformat(),
    }, indent=2))
    print(f"Saved to {config.BEST_PARAMS}. Re-run scripts.backtest_gbm to get the new reported score.")


if __name__ == "__main__":
    main()
