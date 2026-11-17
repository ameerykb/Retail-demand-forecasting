import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum|actual|.
    Preferred over MAPE in retail because it is safe when actuals are 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())
