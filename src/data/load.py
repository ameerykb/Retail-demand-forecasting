from pathlib import Path
import pandas as pd
from .synthetic import make_synthetic_sales


def load_sales(path: str | None = None) -> pd.DataFrame:
    if path and Path(path).exists():
        df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
        df = df.rename(columns={"Date": "date", "Store": "store",
                                "Sales": "sales", "Promo": "promo"})
        return df[["date", "store", "sales", "promo"]].sort_values(["store", "date"])
    return make_synthetic_sales()


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = dates, columns = stores. Makes time-series slicing easy."""
    wide = df.pivot_table(index="date", columns="store", values="sales")
    return wide.asfreq("D").fillna(0)