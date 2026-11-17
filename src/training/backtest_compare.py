"""Backtest that works for ANY learner (LightGBM, PyTorch MLP, ...).

A learner is a function (train_df, test_df, feature_cols) -> predictions for
test_df rows. Baselines are added automatically. Scoring: open days only.
"""
import pandas as pd
from src.training.backtest_gbm import BASELINES, _score


def rolling_backtest_models(df, wide, feature_cols, learners: dict,
                            horizon=42, n_folds=4):
    dates, n, rows = wide.index, len(wide), []
    for fold in range(n_folds):
        c = n - horizon * (n_folds - fold)
        cutoff, end = dates[c], dates[c + horizon - 1]
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] <= end)].copy()

        for name, learner in learners.items():
            test[name] = learner(train, test, feature_cols)

        history = wide.iloc[:c]
        for name, fn in BASELINES.items():
            arr = fn(history, horizon)
            pred = (pd.DataFrame(arr, index=dates[c:c + horizon], columns=history.columns)
                      .rename_axis("date").reset_index()
                      .melt(id_vars="date", var_name="store", value_name=name))
            test = test.merge(pred, on=["date", "store"], how="left")

        for name in [*learners, *BASELINES]:
            rows.append({"fold": fold, "model": name, "cutoff": cutoff.date(),
                         **_score(test, name)})
    return pd.DataFrame(rows)
