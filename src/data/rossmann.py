from pathlib import Path
import pandas as pd


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


def load_rossmann(train_path: str, store_path: str | None = None) -> pd.DataFrame:
    train = pd.read_csv(train_path, parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(store_path) if store_path and Path(store_path).exists() else None
    return standardize_rossmann(train, store)
