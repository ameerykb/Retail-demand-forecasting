"""Rolling backtest that compares LightGBM and the baselines on equal terms:
same folds, same forecast window, scored on OPEN days only."""
import pandas as pd
from src.metrics import mae, rmse, wape
from src.models.baselines import naive, seasonal_naive, moving_average
from src.models.gbm import train_lgbm, predict_lgbm

BASELINES = {"naive": naive, "seasonal_naive": seasonal_naive,
             "moving_average": moving_average}


def _score(frame, col):
    m = frame[frame["open"] == 1]
    return {"MAE": mae(m["sales"], m[col]), "RMSE": rmse(m["sales"], m[col]),
            "WAPE": wape(m["sales"], m[col])}


def rolling_backtest_gbm(df, wide, feature_cols, horizon=42, n_folds=4, params=None):
    dates, n, rows, model = wide.index, len(wide), [], None
    for fold in range(n_folds):
        c = n - horizon * (n_folds - fold)            # position of the cutoff date
        cutoff, end = dates[c], dates[c + horizon - 1]
        train = df[df["date"] < cutoff]
        test = df[(df["date"] >= cutoff) & (df["date"] <= end)].copy()

        model = train_lgbm(train, feature_cols, params)
        test["lightgbm"] = predict_lgbm(model, test, feature_cols)

        history = wide.iloc[:c]                        # baselines see only the past
        for name, fn in BASELINES.items():
            arr = fn(history, horizon)
            pred = (pd.DataFrame(arr, index=dates[c:c + horizon], columns=history.columns)
                      .rename_axis("date").reset_index()
                      .melt(id_vars="date", var_name="store", value_name=name))
            test = test.merge(pred, on=["date", "store"], how="left")

        for name in ["lightgbm", *BASELINES]:
            rows.append({"fold": fold, "model": name, "cutoff": cutoff.date(),
                         **_score(test, name)})
    return pd.DataFrame(rows), model
