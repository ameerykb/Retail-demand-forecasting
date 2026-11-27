"""LightGBM forecaster.

- Trained only on days the store was OPEN (closed days are known in advance
  and are simply predicted as 0).
- Target is log(1 + sales): sales are skewed, and errors in % terms matter more
  than errors in absolute euros.
- With `quantile=q` the same model predicts the q-th quantile instead of the
  mean, which gives prediction intervals (e.g. q=0.1 and q=0.9 -> an 80% range).
  Quantiles survive the log transform because log is monotonic.
"""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=42, verbose=-1, n_jobs=-1)


def load_params(path: str | Path | None) -> dict:
    """Tuned parameters from scripts/tune_lgbm.py if the file exists, else the defaults."""
    if path and Path(path).exists():
        return {**DEFAULT_PARAMS, **json.loads(Path(path).read_text())["params"]}
    return dict(DEFAULT_PARAMS)


def train_lgbm(train, feature_cols, params=None, quantile: float | None = None):
    t = train[train["open"] == 1]
    p = {**DEFAULT_PARAMS, **(params or {})}
    if quantile is not None:
        p.update(objective="quantile", alpha=quantile)
    model = lgb.LGBMRegressor(**p)
    model.fit(t[feature_cols], np.log1p(t["sales"]))
    return model


def predict_lgbm(model, df, feature_cols):
    pred = np.expm1(model.predict(df[feature_cols])).clip(min=0)
    return np.where(df["open"] == 1, pred, 0.0)
