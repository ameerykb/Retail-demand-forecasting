"""Calibrated prediction intervals (conformalized quantile regression, CQR).

Quantile models alone are often too narrow: an "80%" range may only contain 60%
of real values. CQR fixes this with a held-out calibration window:

1. Train the low/high quantile models on data BEFORE the calibration window.
2. On the window, measure how far each real value falls outside its range
   (negative if inside). Work in log space, so the correction is a percentage
   and fits small and large stores alike.
3. The offset is the (1 - alpha) quantile of those scores. Widening every range by
   it gives ~80% coverage on new data (exact in theory if the data are exchangeable).
"""
import numpy as np
import pandas as pd
from src.models.gbm import predict_lgbm, train_lgbm


def conformity_scores(y, lower, upper) -> np.ndarray:
    ly, llo, lhi = (np.log1p(np.asarray(a, dtype=float)) for a in (y, lower, upper))
    return np.maximum(llo - ly, ly - lhi)


def conformal_offset(y, lower, upper, coverage: float) -> float:
    s = np.sort(conformity_scores(y, lower, upper))
    n = len(s)
    k = min(int(np.ceil((n + 1) * coverage)), n) - 1        # finite-sample correction
    return float(s[k])


def widen(lower, upper, offset: float):
    """Apply the offset (in log space). Zeros stay zero, so closed days stay at 0."""
    lo = np.expm1(np.log1p(np.asarray(lower, dtype=float)) - offset).clip(min=0)
    hi = np.expm1(np.log1p(np.asarray(upper, dtype=float)) + offset)
    hi = np.where(np.asarray(upper) == 0, 0.0, hi)
    lo = np.where(np.asarray(lower) == 0, 0.0, lo)
    return lo, hi


def calibrate(train: pd.DataFrame, feature_cols, quantiles, params=None,
              calib_days: int = 42) -> float:
    """Offset learned on the last `calib_days` of `train` (open days only)."""
    q_lo, q_hi = quantiles
    split = train["date"].max() - pd.Timedelta(days=calib_days - 1)
    fit, cal = train[train["date"] < split], train[(train["date"] >= split) & (train["open"] == 1)]
    lo = predict_lgbm(train_lgbm(fit, feature_cols, params, quantile=q_lo), cal, feature_cols)
    hi = predict_lgbm(train_lgbm(fit, feature_cols, params, quantile=q_hi), cal, feature_cols)
    return conformal_offset(cal["sales"], lo, hi, coverage=q_hi - q_lo)
