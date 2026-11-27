"""The forecasting logic behind the API (no web code here, so it is easy to test).

How a forecast is made:
1. The caller gives a store and the SCHEDULE for upcoming days (open/closed, promo, holidays).
2. We append those future rows to the stored sales history.
3. We run the SAME feature code used in training (make_features), so training and
   serving cannot drift apart.
4. The LightGBM model predicts; closed days are set to 0. If the bundle has quantile
   models, each day also gets an 80% range (lower, upper), widened by the
   calibration offset stored in meta.json.
"""
import numpy as np
import pandas as pd
from src.api.bundle import load_bundle
from src.features.build import make_features
from src.models.intervals import widen


class ForecastError(Exception):
    pass


class StoreNotFound(ForecastError):
    pass


class DateOutOfRange(ForecastError):
    pass


class Forecaster:
    def __init__(self, booster, meta, history, stores, quantile_boosters=None):
        self.booster, self.meta, self.stores = booster, meta, stores
        self.quantile_boosters = dict(sorted((quantile_boosters or {}).items()))
        self.horizon = int(meta["horizon_days"])
        self.feature_cols = meta["feature_cols"]
        self.last_date = pd.Timestamp(meta["last_history_date"])
        self.store_ids = sorted(int(s) for s in stores["store"])
        self._store_set = set(self.store_ids)
        self.history = history.merge(stores, on="store", how="left")

    @classmethod
    def load(cls, model_dir):
        return cls(*load_bundle(model_dir))

    @property
    def window(self):
        return self.last_date + pd.Timedelta(days=1), self.last_date + pd.Timedelta(days=self.horizon)

    @property
    def interval(self) -> tuple[float, float] | None:
        qs = list(self.quantile_boosters)
        return (qs[0], qs[-1]) if len(qs) >= 2 else None

    def _predict(self, booster, rows) -> np.ndarray:
        pred = np.expm1(booster.predict(rows[self.feature_cols])).clip(min=0)
        return np.where(rows["open"].to_numpy() == 1, pred, 0.0)

    def forecast(self, store: int, days: list[dict]) -> pd.DataFrame:
        if store not in self._store_set:
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

        # Only this store's history is needed: its lags don't depend on other stores.
        history = self.history[self.history["store"] == store]
        combined = pd.concat([history, future], ignore_index=True)
        feats, _ = make_features(combined, self.horizon)
        rows = feats[feats["date"] > self.last_date].sort_values("date")

        out = pd.DataFrame({"date": rows["date"].dt.date.to_numpy(),
                            "predicted_sales": np.round(self._predict(self.booster, rows), 2)})
        if self.interval:
            low = self._predict(self.quantile_boosters[self.interval[0]], rows)
            high = self._predict(self.quantile_boosters[self.interval[1]], rows)
            low, high = widen(low, high, float(self.meta.get("interval_offset_log", 0.0)))
            # quantile models are trained separately, so keep the range ordered around the forecast
            out["lower"] = np.round(np.minimum(low, out["predicted_sales"]), 2)
            out["upper"] = np.round(np.maximum(high, out["predicted_sales"]), 2)
        return out
