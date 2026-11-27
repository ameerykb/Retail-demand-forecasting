import datetime as dt
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.bundle import save_bundle
from src.api.main import create_app
from src.data.rossmann import standardize_rossmann
from src.data.synthetic import make_rossmann_like
from src.features.build import make_features
from src.models.gbm import train_lgbm, predict_lgbm

H = 42
SMALL = {"n_estimators": 40}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """Train on data BEFORE a cutoff, save the bundle, keep the 42 days after it as 'future'."""
    raw = standardize_rossmann(*make_rossmann_like(n_stores=6, n_days=420))
    cutoff = raw["date"].max() - np.timedelta64(H - 1, "D")
    df_all, cols = make_features(raw, horizon=H)
    model = train_lgbm(df_all[df_all["date"] < cutoff], cols, SMALL)
    out = tmp_path_factory.mktemp("bundle") / "forecaster"
    train = df_all[df_all["date"] < cutoff]
    qmodels = {q: train_lgbm(train, cols, SMALL, quantile=q) for q in (0.1, 0.9)}
    save_bundle(model, cols, raw[raw["date"] < cutoff], H, out, trained_on="synthetic-test",
                backtest_wape=0.09, quantile_models=qmodels)
    return {"dir": str(out), "raw": raw, "df_all": df_all, "cols": cols,
            "model": model, "cutoff": cutoff}


@pytest.fixture(scope="module")
def client(setup):
    with TestClient(create_app(setup["dir"])) as c:
        yield c


def _days(setup, store, n=None):
    raw = setup["raw"]
    w = raw[(raw["store"] == store) & (raw["date"] >= setup["cutoff"])].sort_values("date")
    w = w.head(n) if n else w
    return [{"date": r.date.date().isoformat(), "is_open": int(r.open), "promo": int(r.promo),
             "state_holiday": int(r.state_holiday), "school_holiday": int(r.school_holiday)}
            for r in w.itertuples()]


def test_health_and_model_info(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["horizon_days"] == H and info["n_stores"] == 6
    assert info["backtest_wape"] == 0.09 and "lag_42" in info["features"]


def test_forecast_matches_offline_prediction(client, setup):
    """Serving must give the same numbers as the offline pipeline (no train/serve skew)."""
    days = _days(setup, store=3)
    resp = client.post("/forecast", json={"store": 3, "days": days})
    assert resp.status_code == 200
    served = np.array([f["predicted_sales"] for f in resp.json()["forecasts"]])

    df = setup["df_all"]
    rows = df[(df["store"] == 3) & (df["date"] >= setup["cutoff"])].sort_values("date")
    offline = predict_lgbm(setup["model"], rows, setup["cols"])
    assert len(served) == len(offline) == H
    assert np.allclose(served, offline, atol=0.01)


def test_closed_day_is_zero_and_values_nonnegative(client, setup):
    days = _days(setup, store=2, n=7)
    days[0]["is_open"] = 0
    out = client.post("/forecast", json={"store": 2, "days": days}).json()["forecasts"]
    assert out[0]["predicted_sales"] == 0
    assert all(f["predicted_sales"] >= 0 for f in out)


def test_unknown_store_is_404(client, setup):
    r = client.post("/forecast", json={"store": 999, "days": _days(setup, 1, 3)})
    assert r.status_code == 404


def test_date_out_of_range_is_422(client, setup):
    days = _days(setup, 1, 2)
    days[0]["date"] = (dt.date.fromisoformat(days[0]["date"]) + dt.timedelta(days=H + 5)).isoformat()
    r = client.post("/forecast", json={"store": 1, "days": days})
    assert r.status_code == 422 and "between" in r.json()["detail"]
    old = [{"date": "2020-01-01"}]
    assert client.post("/forecast", json={"store": 1, "days": old}).status_code == 422


def test_invalid_input_is_422(client, setup):
    days = _days(setup, 1, 2)
    bad_flag = [{**days[0], "is_open": 2}]
    assert client.post("/forecast", json={"store": 1, "days": bad_flag}).status_code == 422
    dup = [days[0], days[0]]
    assert client.post("/forecast", json={"store": 1, "days": dup}).status_code == 422
    assert client.post("/forecast", json={"store": 1, "days": []}).status_code == 422
    assert client.post("/forecast", json={"days": days}).status_code == 422


def test_default_endpoint(client):
    r = client.get("/forecast/1/default?n_days=10")
    assert r.status_code == 200 and len(r.json()["forecasts"]) == 10
    assert client.get("/forecast/1/default?n_days=99").status_code == 422
    assert client.get("/forecast/999/default").status_code == 404


def test_app_without_model_returns_503(tmp_path):
    with TestClient(create_app(str(tmp_path / "missing"))) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/forecast/1/default").status_code == 503


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/docs"


def test_model_info_lists_stores_and_interval(client):
    info = client.get("/model-info").json()
    assert info["store_ids"] == [1, 2, 3, 4, 5, 6]
    assert info["interval"] == [0.1, 0.9]


def test_forecast_has_ordered_interval(client, setup):
    out = client.post("/forecast", json={"store": 4, "days": _days(setup, 4)}).json()["forecasts"]
    for f in out:
        assert f["lower"] <= f["predicted_sales"] <= f["upper"]
        if f["predicted_sales"] == 0:
            assert f["lower"] == f["upper"] == 0


def test_requests_are_logged_as_json(client, caplog):
    import json
    import logging
    logger = logging.getLogger("demand_forecast_api")
    logger.addHandler(caplog.handler)
    try:
        client.get("/health")
    finally:
        logger.removeHandler(caplog.handler)
    entry = json.loads(caplog.records[-1].getMessage())
    assert entry["path"] == "/health" and entry["status"] == 200 and entry["duration_ms"] >= 0


def test_bundle_without_intervals_still_works(setup, tmp_path):
    out = save_bundle(setup["model"], setup["cols"], setup["raw"][setup["raw"]["date"] < setup["cutoff"]],
                      H, tmp_path / "plain")
    with TestClient(create_app(str(out))) as c:
        assert c.get("/model-info").json()["interval"] is None
        f = c.get("/forecast/1/default?n_days=3").json()["forecasts"][0]
        assert f["lower"] is None and f["upper"] is None
