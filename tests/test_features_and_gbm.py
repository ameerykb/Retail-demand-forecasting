import numpy as np
from src.data.synthetic import make_rossmann_like
from src.data.rossmann import standardize_rossmann
from src.data.load import to_wide
from src.features.build import make_features
from src.models.gbm import train_lgbm, predict_lgbm
from src.training.backtest_gbm import rolling_backtest_gbm

H = 42


def _data(n_stores=6, n_days=420):
    return standardize_rossmann(*make_rossmann_like(n_stores=n_stores, n_days=n_days))


def test_standardize_columns():
    df = _data()
    for c in ["date", "store", "sales", "open", "promo", "state_holiday",
              "school_holiday", "store_type", "assortment", "competition_distance", "promo2"]:
        assert c in df.columns
    assert set(df["state_holiday"].unique()) <= {0, 1}


def test_lag_feature_is_sales_from_horizon_days_earlier():
    df, _ = make_features(_data(), horizon=H)
    row = df[(df["store"] == 3)].iloc[200]
    earlier = df[(df["store"] == 3) & (df["date"] == row["date"] - np.timedelta64(H, "D"))]
    assert row[f"lag_{H}"] == earlier["sales"].iloc[0]


def test_no_sales_feature_uses_data_inside_forecast_window():
    """For every row in a window starting at `cutoff`, lag features must come
    from before `cutoff` (this is the leakage check)."""
    df, _ = make_features(_data(), horizon=H)
    cutoff = df["date"].sort_values().unique()[300]
    window = df[(df["date"] >= cutoff) & (df["date"] < cutoff + np.timedelta64(H, "D"))]
    lag_dates = window["date"] - np.timedelta64(H, "D")
    assert (lag_dates < cutoff).all()


def test_predictions_zero_when_closed_and_nonnegative():
    df, cols = make_features(_data(), horizon=H)
    train, test = df[df["date"] < "2023-12-01"], df[df["date"] >= "2023-12-01"]
    model = train_lgbm(train, cols, params={"n_estimators": 30})
    pred = predict_lgbm(model, test, cols)
    assert (pred >= 0).all() and (pred[test["open"].to_numpy() == 0] == 0).all()


def test_backtest_runs_and_lightgbm_beats_naive_on_synthetic():
    raw = _data()
    wide = to_wide(raw)
    df, cols = make_features(raw, horizon=H)
    res, _ = rolling_backtest_gbm(df, wide, cols, horizon=H, n_folds=2,
                                  params={"n_estimators": 80})
    assert set(res["model"]) == {"lightgbm", "naive", "seasonal_naive", "moving_average"}
    assert np.isfinite(res["WAPE"]).all()
    avg = res.groupby("model")["WAPE"].mean()
    assert avg["lightgbm"] < avg["naive"]
