import numpy as np
import pandas as pd
from src.metrics import mae, rmse, wape
from src.models.baselines import naive, seasonal_naive, moving_average
from src.training.backtest import rolling_backtest
from src.data.load import load_sales, to_wide


def _wide(n=60):
    idx = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame({1: np.arange(n, dtype=float)}, index=idx)


def test_metrics_perfect_forecast_is_zero():
    y = [1, 2, 3]
    assert mae(y, y) == 0 and rmse(y, y) == 0 and wape(y, y) == 0


def test_wape_value():
    assert wape([10, 10], [8, 12]) == 0.2


def test_naive_repeats_last_value():
    out = naive(_wide(10), 3)
    assert out.shape == (3, 1) and (out == 9).all()


def test_seasonal_naive_repeats_last_week():
    h = _wide(14)
    out = seasonal_naive(h, 7)
    assert (out[:, 0] == np.arange(7, 14)).all()


def test_moving_average_shape():
    assert moving_average(_wide(40), 5, window=7).shape == (5, 1)


def test_backtest_uses_only_past_data():
    wide = _wide(100)
    spy = {}
    def model(history, horizon):
        spy.setdefault("max_seen", []).append(history.index.max())
        return np.zeros((horizon, 1))
    res = rolling_backtest(wide, {"m": model}, horizon=10, n_folds=3)
    assert len(res) == 3
    for seen, cutoff in zip(spy["max_seen"], res["cutoff"]):
        assert seen.date() < cutoff


def test_synthetic_data_shape():
    wide = to_wide(load_sales())
    assert wide.shape == (730, 5) and not wide.isna().any().any()
