"""Makes 4 EDA charts in reports/figures/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.rossmann import load_rossmann, standardize_rossmann
from src.data.synthetic import make_rossmann_like

if Path("data/train.csv").exists():
    df = load_rossmann("data/train.csv", "data/store.csv")
else:
    df = standardize_rossmann(*make_rossmann_like())
out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)
op = df[df["open"] == 1]                                   # open days only

def bar(series, title, fname, labels=None):
    ax = series.plot(kind="bar", figsize=(7, 4), title=title)
    ax.set_ylabel("average daily sales")
    if labels:
        ax.set_xticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.savefig(out / fname, dpi=120)
    plt.close()

bar(op.groupby(op["date"].dt.day_of_week)["sales"].mean(), "Average sales by weekday (open days)",
    "sales_by_weekday.png", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:op["date"].dt.day_of_week.nunique()])
bar(op.groupby(op["date"].dt.month)["sales"].mean(), "Average sales by month", "sales_by_month.png")
bar(op.groupby("promo")["sales"].mean(), "Sales with vs without promo", "sales_by_promo.png", ["No promo", "Promo"])
if "store_type" in op.columns:
    bar(op.groupby("store_type")["sales"].mean(), "Average sales by store type", "sales_by_store_type.png")
op.groupby("date")["sales"].sum().plot(figsize=(9, 3.5), title="Total daily sales (open stores)")
plt.tight_layout()
plt.savefig(out / "total_sales_over_time.png", dpi=120)
plt.close()
print("Saved charts to", out)
