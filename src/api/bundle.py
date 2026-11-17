"""Save / load everything the API needs, in simple, version-proof file formats:

models/forecaster/
  model.txt     LightGBM model in its native text format
  history.csv   last ~130 days of sales for every store (needed to compute lag features)
  stores.csv    one row of attributes per store
  meta.json     feature list, horizon, last history date, version, ...
"""
import json
from pathlib import Path
import lightgbm as lgb
import pandas as pd
from src.features.build import STORE_ATTRS

HISTORY_COLS = ["date", "store", "sales", "open", "promo", "state_holiday", "school_holiday"]


def save_bundle(model, feature_cols, raw: pd.DataFrame, horizon: int, out_dir,
                trained_on: str = "unknown", backtest_wape: float | None = None,
                history_days: int = 130) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))

    last_date = raw["date"].max()
    hist = raw[raw["date"] > last_date - pd.Timedelta(days=history_days)][HISTORY_COLS]
    hist.to_csv(out / "history.csv", index=False)

    attrs = [a for a in STORE_ATTRS if a in raw.columns]
    (raw[["store", *attrs]].drop_duplicates("store").sort_values("store")
        .to_csv(out / "stores.csv", index=False))

    meta = {"model_version": f"lightgbm-{last_date:%Y%m%d}", "algorithm": "lightgbm",
            "horizon_days": horizon, "last_history_date": f"{last_date:%Y-%m-%d}",
            "feature_cols": list(feature_cols), "trained_on": trained_on,
            "backtest_wape": backtest_wape}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_bundle(model_dir):
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(d / "model.txt"))
    history = pd.read_csv(d / "history.csv", parse_dates=["date"])
    stores = pd.read_csv(d / "stores.csv")
    return booster, meta, history, stores
