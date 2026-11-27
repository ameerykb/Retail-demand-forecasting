"""Exploratory charts, saved to reports/figures/.

    python -m scripts.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from src import config  # noqa: E402
from src.data.rossmann import load_data  # noqa: E402


def bar(series, title, fname, labels=None):
    ax = series.plot(kind="bar", figsize=(7, 4), title=title)
    ax.set_ylabel("average daily sales")
    if labels:
        ax.set_xticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.savefig(config.FIGURES / fname, dpi=120)
    plt.close()


def main():
    df, _ = load_data()
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    op = df[df["open"] == 1]                                   # open days only
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    bar(op.groupby(op["date"].dt.day_of_week)["sales"].mean(),
        "Average sales by weekday (open days)", "sales_by_weekday.png",
        days[:op["date"].dt.day_of_week.nunique()])
    bar(op.groupby(op["date"].dt.month)["sales"].mean(), "Average sales by month",
        "sales_by_month.png")
    bar(op.groupby("promo")["sales"].mean(), "Sales with vs without promo",
        "sales_by_promo.png", ["No promo", "Promo"])
    if "store_type" in op.columns:
        bar(op.groupby("store_type")["sales"].mean(), "Average sales by store type",
            "sales_by_store_type.png")
    op.groupby("date")["sales"].sum().plot(figsize=(9, 3.5), title="Total daily sales (open stores)")
    plt.tight_layout()
    plt.savefig(config.FIGURES / "total_sales_over_time.png", dpi=120)
    plt.close()
    print("Saved charts to", config.FIGURES)


if __name__ == "__main__":
    main()
