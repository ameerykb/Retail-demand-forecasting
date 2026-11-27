"""Load the Kaggle Rossmann data (or a synthetic stand-in) in one standard format."""
from pathlib import Path
import pandas as pd
from src import config
from src.data.synthetic import make_rossmann_like


def standardize_rossmann(train: pd.DataFrame, store: pd.DataFrame | None = None) -> pd.DataFrame:
    df = train.rename(columns={"Date": "date", "Store": "store", "Sales": "sales",
                               "Open": "open", "Promo": "promo",
                               "SchoolHoliday": "school_holiday"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["open"] = df["open"].fillna(1).astype(int)
    # StateHoliday is a messy column: "0" means none, "a"/"b"/"c" are holiday types
    df["state_holiday"] = (df["StateHoliday"].astype(str) != "0").astype(int)
    df = df[["date", "store", "sales", "open", "promo", "state_holiday", "school_holiday"]]

    if store is not None:
        st = store.rename(columns={"Store": "store", "Promo2": "promo2"}).copy()
        st["store_type"] = st["StoreType"].astype("category").cat.codes
        st["assortment"] = st["Assortment"].astype("category").cat.codes
        st["competition_distance"] = st["CompetitionDistance"].fillna(
            st["CompetitionDistance"].median())
        df = df.merge(st[["store", "store_type", "assortment",
                          "competition_distance", "promo2"]], on="store", how="left")
    return df.sort_values(["store", "date"]).reset_index(drop=True)


def load_rossmann(train_path: str | Path, store_path: str | Path | None = None) -> pd.DataFrame:
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path) if store_path and Path(store_path).exists() else None
    return standardize_rossmann(train, store)


def load_data(synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    """Real Kaggle data if data/train.csv exists, otherwise synthetic data in the same
    format, so every script (and CI) runs without the download. Returns (df, source)."""
    if config.TRAIN_CSV.exists() and not synthetic:
        return load_rossmann(config.TRAIN_CSV, config.STORE_CSV), "Rossmann Store Sales (Kaggle)"
    return standardize_rossmann(*make_rossmann_like(n_stores=20, n_days=700)), "synthetic"


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = dates, columns = stores (daily grid, gaps filled with 0). Used by the baselines."""
    wide = df.pivot_table(index="date", columns="store", values="sales")
    return wide.asfreq("D").fillna(0)
