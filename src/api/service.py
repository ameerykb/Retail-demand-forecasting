"""The forecasting logic behind the API (no web code here, so it is easy to test).

How a forecast is made:
1. The caller gives a store and the SCHEDULE for upcoming days (open/closed, promo, holidays).
2. We append those future rows to the stored sales history.
3. We run the SAME feature code used in training (make_features), so training and
   serving cannot drift apart.
4. The LightGBM model predicts; closed days are set to 0.
"""
import numpy as np
import pandas as pd
from src.api.bundle import load_bundle
from src.features.build import make_features


class ForecastError(Exception):
    pass


class StoreNotFound(ForecastError):
    pass


class DateOutOfRange(ForecastError):
    pass


class Forecaster:
    def __init__(self, booster, meta, history, stores):
        self.booster, self.meta, self.stores = booster, meta, stores
        self.horizon = int(meta["horizon_days"])
        self.feature_cols = meta["feature_cols"]
        self.last_date = pd.Timestamp(meta["last_history_date"])
        self.store_ids = set(stores["store"].astype(int))
        self.history = history.merge(stores, on="store", how="left")

    @classmethod
    def load(cls, model_dir):
        return cls(*load_bundle(model_dir))

    @property
    def window(self):
        return self.last_date + pd.Timedelta(days=1), self.last_date + pd.Timedelta(days=self.horizon)

    def forecast(self, store: int, days: list[dict]) -> pd.DataFrame:
        if store not in self.store_ids:
            raise StoreNotFound(f"Store {store} is not known to this model.")
        lo, hi = self.window
        dates = [pd.Timestamp(d["date"]) for d in days]
        bad = [d.date().isoformat() for d in dates if not (lo <= d <= hi)]
        if bad:
            raise DateOutOfRange(
                f"Dates must be between {lo.date()} and {hi.date()} "
                f"(1 to {self.horizon} days after the last history date). Out of range: {bad[:3]}")

        future = pd.DataFrame({
            "date": dates, "store": store, "sales": np.nan,
            "open": [d["open"] for d in days], "promo": [d["promo"] for d in days],
            "state_holiday": [d["state_holiday"] for d in days],
            "school_holiday": [d["school_holiday"] for d in days]})
        future = future.merge(self.stores, on="store", how="left")

        combined = pd.concat([self.history, future], ignore_index=True)
        feats, _ = make_features(combined, self.horizon)
        rows = feats[(feats["store"] == store) & (feats["date"] > self.last_date)].sort_values("date")

        raw = self.booster.predict(rows[self.feature_cols])
        pred = np.expm1(raw).clip(min=0)
        pred = np.where(rows["open"].to_numpy() == 1, pred, 0.0)
        return pd.DataFrame({"date": rows["date"].dt.date.to_numpy(),
                             "predicted_sales": np.round(pred, 2)})
