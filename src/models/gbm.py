"""LightGBM forecaster.

- Trained only on days the store was OPEN (closed days are known in advance
  and are simply predicted as 0).
- Target is log(1 + sales): sales are skewed, and errors in % terms matter more
  than errors in absolute euros.
"""
import numpy as np
import lightgbm as lgb

DEFAULT_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      random_state=42, verbose=-1, n_jobs=-1)


def train_lgbm(train, feature_cols, params=None):
    t = train[train["open"] == 1]
    model = lgb.LGBMRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(t[feature_cols], np.log1p(t["sales"]))
    return model


def predict_lgbm(model, df, feature_cols):
    pred = np.expm1(model.predict(df[feature_cols])).clip(min=0)
    return np.where(df["open"] == 1, pred, 0.0)
