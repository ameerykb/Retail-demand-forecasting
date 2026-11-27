import mlflow

DEFAULT_URI = "sqlite:///mlflow.db"   # a local file in the project folder


def log_results_to_mlflow(results, params_by_model=None, curves_by_model=None,
                          shared_params=None, experiment="demand-forecasting",
                          tracking_uri=DEFAULT_URI):
    params_by_model = params_by_model or {}
    curves_by_model = curves_by_model or {}
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    for model, g in results.groupby("model"):
        with mlflow.start_run(run_name=model):
            mlflow.log_param("model", model)
            mlflow.log_params({**(shared_params or {}), **params_by_model.get(model, {})})
            for _, r in g.iterrows():                       # one point per fold
                mlflow.log_metrics({"fold_MAE": r["MAE"], "fold_RMSE": r["RMSE"],
                                    "fold_WAPE": r["WAPE"]}, step=int(r["fold"]))
            mlflow.log_metrics({"MAE": g["MAE"].mean(), "RMSE": g["RMSE"].mean(),
                                "WAPE": g["WAPE"].mean()})   # averages across folds
            for fold, curve in enumerate(curves_by_model.get(model, [])):
                for epoch, (tl, vl) in enumerate(zip(curve["train_loss"], curve["val_loss"], strict=True)):
                    mlflow.log_metric(f"fold{fold}_train_loss", tl, step=epoch)
                    mlflow.log_metric(f"fold{fold}_val_loss", vl, step=epoch)
