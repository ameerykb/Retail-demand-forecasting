"""Baseline forecasters. Every model takes (history, horizon) and returns
an array of shape (horizon, n_stores). Any fancy model must beat these."""
import numpy as np
import pandas as pd


def naive(history: pd.DataFrame, horizon: int) -> np.ndarray:
    """Tomorrow = today. Repeat the last observed value."""
    return np.tile(history.iloc[-1].to_numpy(), (horizon, 1))


def seasonal_naive(history: pd.DataFrame, horizon: int, season: int = 7) -> np.ndarray:
    """Same weekday last week. Strong baseline for retail data."""
    last_season = history.iloc[-season:].to_numpy()
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, (reps, 1))[:horizon]


def moving_average(history: pd.DataFrame, horizon: int, window: int = 28) -> np.ndarray:
    """Average of the last `window` days, repeated."""
    return np.tile(history.iloc[-window:].mean().to_numpy(), (horizon, 1))
