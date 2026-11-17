from src.data.load import load_sales, to_wide
from src.models.baselines import naive, seasonal_naive, moving_average
from src.training.backtest import rolling_backtest, summarize

df = load_sales("data/train.csv")
wide = to_wide(df)
print(f"Data: {wide.shape[0]} days x {wide.shape[1]} stores "
      f"({wide.index.min().date()} to {wide.index.max().date()})")

models = {"naive": naive, "seasonal_naive": seasonal_naive, "moving_average": moving_average}
results = rolling_backtest(wide, models, horizon=28, n_folds=4)
print("\nPer-fold results:\n", results.to_string(index=False))
print("\nAverage across folds (lower is better):\n", summarize(results))
