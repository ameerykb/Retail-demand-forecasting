"""Which features drive the forecast? SHAP values from LightGBM itself
(pred_contrib=True gives exact TreeSHAP contributions, no extra library needed).

    python -m scripts.explain          # saves reports/figures/shap_importance.png

The model is trained on everything before the last HORIZON days and explained on
those last days (open stores only), i.e. on data it has not seen.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from src import config  # noqa: E402
from src.data.rossmann import load_data  # noqa: E402
from src.features.build import make_features  # noqa: E402
from src.models.gbm import load_params, train_lgbm  # noqa: E402

LABELS = {
    "store_cat": "Store (identity)", "dow": "Day of week", "month": "Month", "day": "Day of month",
    "weekofyear": "Week of year", "dayofyear": "Day of year", "promo": "Promotion",
    "state_holiday": "State holiday", "school_holiday": "School holiday",
    "store_type": "Store type", "assortment": "Assortment",
    "competition_distance": "Competitor distance", "promo2": "Promo2 (loyalty)",
    "roll_mean_28": "Sales, 28-day mean (42+ days ago)",
    "roll_std_28": "Sales, 28-day std (42+ days ago)",
}


def label(col: str) -> str:
    if col.startswith("lag_"):
        return f"Sales {col.split('_')[1]} days ago"
    return LABELS.get(col, col)


def shap_importance(model, X: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, in units of log(1 + sales)."""
    contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]   # last column = bias
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values()


def main(max_rows: int = 50_000):
    raw, source = load_data()
    df, cols = make_features(raw, horizon=config.HORIZON)
    cutoff = df["date"].max() - pd.Timedelta(days=config.HORIZON - 1)
    train, test = df[df["date"] < cutoff], df[(df["date"] >= cutoff) & (df["open"] == 1)]
    model = train_lgbm(train, cols, load_params(config.BEST_PARAMS))
    X = test[cols].sample(min(max_rows, len(test)), random_state=42)
    imp = shap_importance(model, X)

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_importance(imp.rename(label), config.FIGURES / "shap_importance.png",
                    f"Data: {source}, last {config.HORIZON} days (not seen in training)")
    print(imp.sort_values(ascending=False).rename(label).round(4).to_string())
    print(f"\nSaved {config.FIGURES / 'shap_importance.png'}")


def plot_importance(imp: pd.Series, out, subtitle: str = ""):
    """Horizontal bar chart of mean |SHAP| (imp is indexed by readable labels)."""
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.to_numpy(), color="#2563eb")
    ax.set_xlabel("mean |SHAP value|  (impact on log sales)")
    fig.suptitle("What drives the forecast", x=0.02, ha="left", fontsize=14, fontweight="bold")
    if subtitle:
        ax.set_title(subtitle, loc="left", fontsize=10, color="#555555")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
