import numpy as np
import torch
import mlflow
from src.data.synthetic import make_rossmann_like
from src.data.rossmann import standardize_rossmann
from src.data.rossmann import to_wide
from src.features.build import make_features
from src.models.gbm import train_lgbm, predict_lgbm
from src.models.nn import Preprocessor, TabularMLP, fit_mlp, make_mlp_learner
from src.training.backtest import rolling_backtest_models
from src.tracking import log_results_to_mlflow

H = 42
SMALL = dict(epochs=4, hidden=(32, 16), batch_size=256)


def _prepared():
    raw = standardize_rossmann(*make_rossmann_like(n_stores=6, n_days=420))
    wide = to_wide(raw)
    df, cols = make_features(raw, horizon=H)
    return df, wide, cols


def test_preprocessor_shapes_and_ranges():
    df, _, cols = _prepared()
    pre = Preprocessor(cols).fit(df)
    cats, nums = pre.transform(df)
    assert cats.shape[0] == nums.shape[0] == len(df)
    assert not torch.isnan(nums).any()
    for i, c in enumerate(pre.cat_cols):
        assert cats[:, i].min() >= 0 and cats[:, i].max() < pre.cardinality[c]


def test_model_forward_shape():
    df, _, cols = _prepared()
    pre = Preprocessor(cols).fit(df)
    cats, nums = pre.transform(df.head(10))
    model = TabularMLP(pre.cardinality, nums.shape[1], hidden=(8,))
    assert model(cats, nums).shape == (10,)


def test_training_reduces_loss_and_predictions_are_valid():
    df, _, cols = _prepared()
    cut = df["date"].max() - np.timedelta64(60, "D")
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    fc = fit_mlp(train, cols, **SMALL)
    assert fc.history["train_loss"][-1] < fc.history["train_loss"][0]
    pred = fc.predict(test)
    assert len(pred) == len(test) and (pred >= 0).all()
    assert (pred[test["open"].to_numpy() == 0] == 0).all()


def test_same_seed_gives_same_predictions():
    df, _, cols = _prepared()
    train = df[df["date"] < df["date"].max() - np.timedelta64(60, "D")]
    a = fit_mlp(train, cols, **SMALL).predict(train.head(50))
    b = fit_mlp(train, cols, **SMALL).predict(train.head(50))
    assert np.allclose(a, b)


def test_compare_backtest_and_mlflow_logging(tmp_path):
    df, wide, cols = _prepared()
    curves = []
    learners = {
        "lightgbm": lambda tr, te, f: predict_lgbm(train_lgbm(tr, f, {"n_estimators": 30}), te, f),
        "mlp_pytorch": make_mlp_learner(curves, **SMALL),
    }
    res = rolling_backtest_models(df, wide, cols, learners, horizon=H, n_folds=2)
    assert set(res["model"]) == {"lightgbm", "mlp_pytorch", "naive",
                                 "seasonal_naive", "moving_average"}
    assert np.isfinite(res["WAPE"]).all() and len(curves) == 2

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    log_results_to_mlflow(res, curves_by_model={"mlp_pytorch": curves},
                          shared_params={"horizon": H}, tracking_uri=uri)
    runs = mlflow.search_runs(experiment_names=["demand-forecasting"])
    assert len(runs) == 5 and "metrics.WAPE" in runs.columns
