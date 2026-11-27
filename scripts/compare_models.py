"""PyTorch MLP vs LightGBM vs baselines, logged to MLflow.

    python -m scripts.compare_models [--epochs 15] [--folds 4]
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
from src import config
from src.data.rossmann import load_data, to_wide
from src.features.build import make_features
from src.models.gbm import load_params, predict_lgbm, train_lgbm
from src.models.nn import MLP_DEFAULTS, make_mlp_learner
from src.tracking import log_results_to_mlflow
from src.training.backtest import rolling_backtest_models, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=MLP_DEFAULTS["epochs"])
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    args = ap.parse_args()

    df, source = load_data()
    print(f"Using {source}: {len(df):,} rows, {df['store'].nunique()} stores")
    wide = to_wide(df)
    df, cols = make_features(df, horizon=config.HORIZON)
    params = load_params(config.BEST_PARAMS)

    mlp_curves = []
    learners = {
        "lightgbm": lambda tr, te, f: predict_lgbm(train_lgbm(tr, f, params), te, f),
        "mlp_pytorch": make_mlp_learner(mlp_curves, epochs=args.epochs),
    }
    results = rolling_backtest_models(df, wide, cols, learners,
                                      horizon=config.HORIZON, n_folds=args.folds)
    print("\nPer-fold results (open days only):\n", results.to_string(index=False))
    print("\nAverage across folds (lower is better):\n", summarize(results))

    log_results_to_mlflow(
        results,
        params_by_model={"lightgbm": params,
                         "mlp_pytorch": {**MLP_DEFAULTS, "epochs": args.epochs}},
        curves_by_model={"mlp_pytorch": mlp_curves},
        shared_params={"data_source": source, "horizon": config.HORIZON, "n_folds": args.folds})
    print("\nLogged to MLflow. View it with:\n  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
