"""Synthetic daily store-sales data.

Why this exists: lets the whole pipeline (and CI tests) run without
downloading a dataset. It has the same patterns real retail data has:
weekly seasonality, yearly seasonality, a trend, promotions and noise.
Swap in the real Rossmann data later via src/data/load.py.
"""
import numpy as np
import pandas as pd


def make_synthetic_sales(n_stores: int = 5, n_days: int = 730,
                         start: str = "2023-01-01", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)  # fixed seed -> reproducible data
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for store in range(1, n_stores + 1):
        base = rng.uniform(800, 2000)                 # store size
        trend = np.linspace(0, rng.uniform(0.0, 0.3), n_days)
        weekly = np.array([0.9, 0.95, 1.0, 1.0, 1.1, 1.3, 0.7])[dates.day_of_week]
        yearly = 1 + 0.15 * np.sin(2 * np.pi * dates.day_of_year / 365.25)
        promo = (rng.random(n_days) < 0.2).astype(int)  # ~20% of days on promo
        promo_effect = 1 + 0.25 * promo
        noise = rng.normal(1.0, 0.05, n_days)
        sales = base * (1 + trend) * weekly * yearly * promo_effect * noise
        rows.append(pd.DataFrame({"date": dates, "store": store,
                                  "sales": sales.round(2), "promo": promo}))
    return pd.concat(rows, ignore_index=True)


def make_rossmann_like(n_stores: int = 8, n_days: int = 500, seed: int = 42):
    """Fake data in the exact column format of Rossmann's train.csv / store.csv.
    Used by tests and as a fallback so the pipeline runs without the real files."""
    df = make_synthetic_sales(n_stores=n_stores, n_days=n_days, seed=seed)
    rng = np.random.default_rng(seed)
    closed = df["date"].dt.day_of_week == 6            # closed on Sundays
    sales = np.where(closed, 0.0, df["sales"])
    train = pd.DataFrame({
        "Store": df["store"], "DayOfWeek": df["date"].dt.day_of_week + 1,
        "Date": df["date"], "Sales": sales, "Customers": (sales / 9).round(),
        "Open": (~closed).astype(int), "Promo": df["promo"],
        "StateHoliday": np.where(rng.random(len(df)) < 0.02, "a", "0"),
        "SchoolHoliday": (rng.random(len(df)) < 0.15).astype(int)})
    store = pd.DataFrame({
        "Store": np.arange(1, n_stores + 1),
        "StoreType": rng.choice(list("abcd"), n_stores),
        "Assortment": rng.choice(list("abc"), n_stores),
        "CompetitionDistance": rng.uniform(100, 20000, n_stores),
        "Promo2": rng.integers(0, 2, n_stores)})
    return train, store
