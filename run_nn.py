
import argparse
from pathlib import Path
from src.data.rossmann import load_rossmann, standardize_rossmann
from src.data.synthetic import make_rossmann_like
from src.data.load import to_wide
from src.features.build import make_features
from src.models.gbm import train_lgbm, predict_lgbm, DEFAULT_PARAMS
from src.models.nn import make_mlp_learner, MLP_DEFAULTS
from src.training.backtest import summarize
from src.training.backtest_compare import rolling_backtest_models
from src.tracking import log_results_to_mlflow

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=MLP_DEFAULTS["epochs"])
ap.add_argument("--folds", type=int, default=4)
args = ap.parse_args()
HORIZON = 42

if Path("data/train.csv").exists():
    df, source = load_rossmann("data/train.csv", "data/store.csv"), "real Rossmann data"
else:
    df, source = standardize_rossmann(*make_rossmann_like(n_stores=20, n_days=700)), "SYNTHETIC data"
print(f"Using {source}: {len(df):,} rows, {df['store'].nunique()} stores")

wide = to_wide(df)
df, feature_cols = make_features(df, horizon=HORIZON)

mlp_curves = []
learners = {
    "lightgbm": lambda tr, te, f: predict_lgbm(train_lgbm(tr, f), te, f),
    "mlp_pytorch": make_mlp_learner(mlp_curves, epochs=args.epochs),
}
results = rolling_backtest_models(df, wide, feature_cols, learners,
                                  horizon=HORIZON, n_folds=args.folds)
print("\nPer-fold results (open days only):\n", results.to_string(index=False))
print("\nAverage across folds (lower is better):\n", summarize(results))

log_results_to_mlflow(
    results,
    params_by_model={"lightgbm": DEFAULT_PARAMS,
                     "mlp_pytorch": {**MLP_DEFAULTS, "epochs": args.epochs}},
    curves_by_model={"mlp_pytorch": mlp_curves},
    shared_params={"data_source": source, "horizon": HORIZON, "n_folds": args.folds})
print("\nLogged to MLflow. View it with:\n  mlflow ui --backend-store-uri sqlite:///mlflow.db")
