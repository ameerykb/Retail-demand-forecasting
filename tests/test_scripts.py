"""Smoke tests for the scripts' building blocks (tuning, explainability, intervals, config)."""
import json
import numpy as np
import optuna
from src import config
from src.data.rossmann import load_data, to_wide
from src.features.build import make_features
from src.models.gbm import DEFAULT_PARAMS, load_params, train_lgbm
from src.training.backtest import backtest_intervals
from scripts.explain import label, shap_importance
from scripts.tune_lgbm import search_space, tuning_data

H = config.HORIZON


def _small():
    raw, _ = load_data(synthetic=True)
    raw = raw[raw["store"] <= 4]
    df, cols = make_features(raw, horizon=H)
    return raw, df, cols


def test_load_params_defaults_and_tuned(tmp_path):
    assert load_params(None) == DEFAULT_PARAMS
    f = tmp_path / "best.json"
    f.write_text(json.dumps({"params": {"num_leaves": 17}}))
    p = load_params(f)
    assert p["num_leaves"] == 17 and p["learning_rate"] == DEFAULT_PARAMS["learning_rate"]


def test_tuning_never_sees_the_reported_period():
    raw, _, _ = _small()
    holdout = config.N_FOLDS * H
    t = tuning_data(raw, holdout)
    first_reported = raw["date"].max() - np.timedelta64(holdout - 1, "D")
    assert t["date"].max() < first_reported


def test_search_space_is_valid():
    p = search_space(optuna.trial.FixedTrial({
        "learning_rate": 0.05, "n_estimators": 400, "num_leaves": 63, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 0.1}))
    assert set(p) >= {"learning_rate", "num_leaves", "n_estimators"}


def test_shap_importance_covers_every_feature():
    _, df, cols = _small()
    model = train_lgbm(df, cols, {"n_estimators": 20})
    imp = shap_importance(model, df[df["open"] == 1][cols].head(300))
    assert set(imp.index) == set(cols) and (imp >= 0).all() and imp.sum() > 0
    assert label("lag_42") == "Sales 42 days ago" and label("promo") == "Promotion"


def test_interval_backtest_reports_coverage():
    raw, df, cols = _small()
    res = backtest_intervals(df, to_wide(raw), cols, (0.1, 0.9), horizon=H, n_folds=2,
                             params={"n_estimators": 40})
    assert len(res) == 2
    assert res[["raw_coverage", "calibrated_coverage"]].stack().between(0, 1).all()
    assert (res["calibrated_width"] >= res["raw_width"] - 1e-9).all() or (res["offset_log"] < 0).any()


def test_quantile_models_are_ordered_on_average():
    _, df, cols = _small()
    lo = train_lgbm(df, cols, {"n_estimators": 40}, quantile=0.1)
    hi = train_lgbm(df, cols, {"n_estimators": 40}, quantile=0.9)
    rows = df[df["open"] == 1][cols]
    assert (hi.predict(rows) - lo.predict(rows)).mean() > 0


def test_conformal_offset_reaches_target_coverage():
    from src.metrics import interval_coverage
    from src.models.intervals import conformal_offset, widen
    rng = np.random.default_rng(0)
    y = rng.lognormal(8, 0.3, 2000)
    lo, hi = y * 0.95, y * 1.05              # deliberately far too narrow
    lo, hi = lo * rng.lognormal(0, 0.2, 2000), hi * rng.lognormal(0, 0.2, 2000)
    q = conformal_offset(y, lo, hi, coverage=0.8)
    clo, chi = widen(lo, hi, q)
    assert interval_coverage(y, lo, hi) < 0.6
    assert 0.79 <= interval_coverage(y, clo, chi) <= 0.82


def test_widen_keeps_closed_days_at_zero():
    from src.models.intervals import widen
    lo, hi = widen(np.array([0.0, 100.0]), np.array([0.0, 200.0]), 0.1)
    assert lo[0] == hi[0] == 0 and lo[1] < 100 and hi[1] > 200
