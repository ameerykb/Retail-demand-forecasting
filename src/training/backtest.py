"""Rolling-origin backtesting.

A single random train/test split is WRONG for time series (it lets the model
see the future). Instead we simulate real use: train on everything before a
cutoff date, forecast the next `horizon` days, score it, then move the cutoff
forward and repeat. Every model sees exactly the same folds.
"""
from collections.abc import Callable
import numpy as np
import pandas as pd
from src.metrics import interval_coverage, mae, rmse, wape
from src.models.baselines import moving_average, naive, seasonal_naive
from src.models.gbm import predict_lgbm, train_lgbm

BASELINES = {"naive": naive, "seasonal_naive": seasonal_naive,
             "moving_average": moving_average}

# A learner: (train_rows, test_rows, feature_cols) -> predictions for test_rows
Learner = Callable[[pd.DataFrame, pd.DataFrame, list[str]], np.ndarray]


def fold_cutoffs(dates: pd.DatetimeIndex, horizon: int, n_folds: int):
    """Yield (fold, start position, cutoff date, last date) for each test window."""
    n = len(dates)
    for fold in range(n_folds):
        c = n - horizon * (n_folds - fold)
        yield fold, c, dates[c], dates[c + horizon - 1]


def score_open_days(frame: pd.DataFrame, col: str) -> dict:
    """Closed days are known in advance and always predicted as 0, so they are not scored."""
    m = frame[frame["open"] == 1]
    return {"MAE": mae(m["sales"], m[col]), "RMSE": rmse(m["sales"], m[col]),
            "WAPE": wape(m["sales"], m[col])}


def rolling_backtest(wide: pd.DataFrame, models: dict[str, Callable],
                     horizon: int = 28, n_folds: int = 4) -> pd.DataFrame:
    """Baseline-style models that only need the date x store sales grid."""
    results = []
    for fold, c, cutoff, _ in fold_cutoffs(wide.index, horizon, n_folds):
        history = wide.iloc[:c]
        actual = wide.iloc[c:c + horizon].to_numpy()
        for name, model in models.items():
            pred = model(history, horizon)
            results.append({"fold": fold, "model": name, "cutoff": cutoff.date(),
                            "MAE": mae(actual, pred), "RMSE": rmse(actual, pred),
                            "WAPE": wape(actual, pred)})
    return pd.DataFrame(results)


def rolling_backtest_models(df: pd.DataFrame, wide: pd.DataFrame, feature_cols: list[str],
                            learners: dict[str, Learner], horizon: int = 42,
                            n_folds: int = 4, baselines: bool = True) -> pd.DataFrame:
    """Backtest for feature-based learners (LightGBM, PyTorch MLP, ...), with the
    baselines added on the same folds. Scored on open days only."""
    dates, rows = wide.index, []
    for fold, c, cutoff, end in fold_cutoffs(dates, horizon, n_folds):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] <= end)].copy()

        for name, learner in learners.items():
            test[name] = learner(train, test, feature_cols)

        names = list(learners)
        if baselines:
            history = wide.iloc[:c]                            # baselines see only the past
            for name, fn in BASELINES.items():
                pred = (pd.DataFrame(fn(history, horizon), index=dates[c:c + horizon],
                                     columns=history.columns)
                        .rename_axis("date").reset_index()
                        .melt(id_vars="date", var_name="store", value_name=name))
                test = test.merge(pred, on=["date", "store"], how="left")
            names += list(BASELINES)

        for name in names:
            rows.append({"fold": fold, "model": name, "cutoff": cutoff.date(),
                         **score_open_days(test, name)})
    return pd.DataFrame(rows)


def rolling_backtest_gbm(df, wide, feature_cols, horizon=42, n_folds=4, params=None):
    """LightGBM vs baselines. Also returns the last fold's model (for feature importance)."""
    last = {}

    def lightgbm(train, test, cols):
        last["model"] = train_lgbm(train, cols, params)
        return predict_lgbm(last["model"], test, cols)

    res = rolling_backtest_models(df, wide, feature_cols, {"lightgbm": lightgbm},
                                  horizon, n_folds)
    return res, last.get("model")


def backtest_intervals(df: pd.DataFrame, wide: pd.DataFrame, feature_cols: list[str],
                       quantiles=(0.1, 0.9), horizon: int = 42, n_folds: int = 4,
                       params: dict | None = None) -> pd.DataFrame:
    """How often does the real value fall inside the predicted [low, high] range?
    For an 80% interval (p10 to p90) a well-calibrated model should be near 0.80.
    Reports the raw quantile models and the conformally calibrated version (as served)."""
    from src.models.intervals import calibrate, widen   # local import: avoids a cycle

    lo_q, hi_q = quantiles
    rows = []
    for fold, _, cutoff, end in fold_cutoffs(wide.index, horizon, n_folds):
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] <= end)]
        test = test[test["open"] == 1]
        offset = calibrate(train, feature_cols, quantiles, params, calib_days=horizon)
        lo = predict_lgbm(train_lgbm(train, feature_cols, params, quantile=lo_q), test, feature_cols)
        hi = predict_lgbm(train_lgbm(train, feature_cols, params, quantile=hi_q), test, feature_cols)
        clo, chi = widen(lo, hi, offset)
        rows.append({"fold": fold, "cutoff": cutoff.date(),
                     "raw_coverage": interval_coverage(test["sales"], lo, hi),
                     "calibrated_coverage": interval_coverage(test["sales"], clo, chi),
                     "raw_width": float(np.mean(hi - lo)),
                     "calibrated_width": float(np.mean(chi - clo)),
                     "offset_log": offset})
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (results.groupby("model")[["MAE", "RMSE", "WAPE"]].mean()
            .sort_values("WAPE").round(4))
