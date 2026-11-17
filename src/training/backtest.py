"""Rolling-origin backtesting.

A single random train/test split is WRONG for time series (it lets the model
see the future). Instead we simulate real use: train on everything before a
cutoff date, forecast the next `horizon` days, score it, then move the cutoff
forward and repeat.
"""
from typing import Callable
import pandas as pd
from src.metrics import mae, rmse, wape


def rolling_backtest(wide: pd.DataFrame, models: dict[str, Callable],
                     horizon: int = 28, n_folds: int = 4) -> pd.DataFrame:
    results = []
    n = len(wide)
    for fold in range(n_folds):
        cutoff = n - horizon * (n_folds - fold)      # index where the test window starts
        history = wide.iloc[:cutoff]
        actual = wide.iloc[cutoff:cutoff + horizon].to_numpy()
        for name, model in models.items():
            pred = model(history, horizon)
            results.append({"fold": fold, "model": name,
                            "cutoff": wide.index[cutoff].date(),
                            "MAE": mae(actual, pred),
                            "RMSE": rmse(actual, pred),
                            "WAPE": wape(actual, pred)})
    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (results.groupby("model")[["MAE", "RMSE", "WAPE"]].mean()
            .sort_values("WAPE").round(4))
