"""Feature engineering for multi-week-ahead forecasting.

KEY IDEA (avoiding data leakage): we forecast `horizon` days ahead, so at
forecast time we only know sales up to the cutoff date. Therefore every
sales-based feature is shifted by AT LEAST `horizon` days. A row for date t
then only uses sales from t - horizon or earlier, which is always before the
cutoff for any t inside the forecast window.

Features known in advance (calendar, promo, holidays, open/closed schedule,
store attributes) are used as-is.
"""
import pandas as pd

STORE_ATTRS = ["store_type", "assortment", "competition_distance", "promo2"]


def make_features(df: pd.DataFrame, horizon: int = 42):
    df = df.copy()
    d = df["date"]
    df["dow"] = d.dt.day_of_week
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["dayofyear"] = d.dt.day_of_year

    # date x store matrix, so lags are by calendar date (not by row position)
    wide = (df.pivot_table(index="date", columns="store", values="sales")
              .asfreq("D").rename_axis("date"))
    # extend the grid to the last date in df, so rows for future dates get lag values too
    wide = wide.reindex(pd.date_range(wide.index.min(), df["date"].max(), freq="D")).rename_axis("date")
    lagged = wide.shift(horizon)
    lag_features = {
        f"lag_{horizon}": lagged,
        f"lag_{horizon + 7}": wide.shift(horizon + 7),
        f"lag_{horizon + 14}": wide.shift(horizon + 14),
        "roll_mean_28": lagged.rolling(28, min_periods=7).mean(),
        "roll_std_28": lagged.rolling(28, min_periods=7).std(),
    }
    for name, w in lag_features.items():
        long = w.reset_index().melt(id_vars="date", var_name="store", value_name=name)
        df = df.merge(long, on=["date", "store"], how="left")

    df["store_cat"] = df["store"].astype("category")
    feature_cols = (["store_cat", "dow", "month", "day", "weekofyear", "dayofyear",
                     "promo", "state_holiday", "school_holiday"]
                    + [a for a in STORE_ATTRS if a in df.columns]
                    + list(lag_features))
    return df, feature_cols
