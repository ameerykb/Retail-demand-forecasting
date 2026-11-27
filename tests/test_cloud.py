"""Tests for the AWS pieces: loading the bundle from S3 and the Lambda handler.
No AWS account needed: S3 is replaced by a fake client that copies local files."""
import importlib
import json
import shutil
from pathlib import Path
import pytest
from src.api import model_store
from src.api.bundle import save_bundle
from src.api.model_store import BUNDLE_FILES, download_bundle, parse_s3_uri
from src.data.rossmann import standardize_rossmann
from src.data.synthetic import make_rossmann_like
from src.features.build import make_features
from src.models.gbm import train_lgbm

H = 42


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    raw = standardize_rossmann(*make_rossmann_like(n_stores=4, n_days=300))
    df, cols = make_features(raw, horizon=H)
    model = train_lgbm(df, cols, {"n_estimators": 20})
    qmodels = {q: train_lgbm(df, cols, {"n_estimators": 20}, quantile=q) for q in (0.1, 0.9)}
    return save_bundle(model, cols, raw, H, tmp_path_factory.mktemp("b") / "forecaster",
                       trained_on="synthetic-test", quantile_models=qmodels)


class FakeS3:
    """Stands in for boto3's S3 client: bucket 'models' maps to a local folder."""
    def __init__(self, root: Path):
        self.root, self.calls = root, []

    def download_file(self, bucket, key, filename):
        self.calls.append((bucket, key))
        shutil.copy(self.root / Path(key).name, filename)


def test_parse_s3_uri():
    assert parse_s3_uri("s3://my-bucket/forecaster/") == ("my-bucket", "forecaster")
    assert parse_s3_uri("s3://my-bucket") == ("my-bucket", "")
    with pytest.raises(ValueError):
        parse_s3_uri("https://example.com/x")
    with pytest.raises(ValueError):
        parse_s3_uri("s3:///no-bucket")


def test_download_bundle_fetches_all_files_then_caches(bundle_dir, tmp_path):
    s3 = FakeS3(bundle_dir)
    dest = tmp_path / "forecaster"
    download_bundle("s3://models/forecaster", dest, client=s3)
    expected = [*BUNDLE_FILES, "model_q10.txt", "model_q90.txt"]
    assert sorted(s3.calls) == sorted(("models", f"forecaster/{f}") for f in expected)
    assert all((dest / f).exists() for f in expected)

    download_bundle("s3://models/forecaster", dest, client=s3)   # warm container
    assert len(s3.calls) == len(expected)                         # no second download


def test_load_forecaster_uses_s3_when_configured(bundle_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_S3_URI", "s3://models/forecaster")
    real = model_store.download_bundle
    monkeypatch.setattr(model_store, "download_bundle",
                        lambda uri, dest: real(uri, dest, client=FakeS3(bundle_dir)))
    fc = model_store.load_forecaster(tmp_path / "forecaster")
    assert fc is not None and fc.horizon == H


def test_load_forecaster_without_model_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_S3_URI", raising=False)
    assert model_store.load_forecaster(tmp_path / "missing") is None


def _http_api_event(method, path, query="", body=None):
    """Minimal API Gateway HTTP API (payload v2.0) event, as Lambda receives it."""
    return {
        "version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": query,
        "headers": {"host": "example.execute-api.ap-south-1.amazonaws.com",
                    "content-type": "application/json"},
        "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1",
                                    "sourceIp": "127.0.0.1", "userAgent": "pytest"},
                           "stage": "$default", "requestId": "test", "accountId": "123"},
        "body": json.dumps(body) if body is not None else None,
        "isBase64Encoded": False,
    }


def test_lambda_handler_serves_forecasts(bundle_dir, monkeypatch):
    monkeypatch.delenv("MODEL_S3_URI", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(bundle_dir))
    import src.api.lambda_handler as lh
    lh = importlib.reload(lh)          # re-run the import-time model load with this MODEL_DIR

    r = lh.handler(_http_api_event("GET", "/health"), None)
    assert r["statusCode"] == 200 and json.loads(r["body"])["model_loaded"] is True

    r = lh.handler(_http_api_event("GET", "/forecast/1/default", "n_days=7"), None)
    assert r["statusCode"] == 200 and len(json.loads(r["body"])["forecasts"]) == 7

    r = lh.handler(_http_api_event("POST", "/forecast", body={"store": 999, "days": []}), None)
    assert r["statusCode"] == 422
