"""Where the API gets its trained model from.

Locally (uvicorn / Docker Compose) the bundle sits in MODEL_DIR on disk.
On AWS Lambda the image contains only code, so the bundle lives in S3:
set MODEL_S3_URI=s3://<bucket>/<prefix> and it is downloaded to MODEL_DIR
(on Lambda that must be under /tmp, the only writable folder).
"""
import json
import os
from pathlib import Path
from src.api.bundle import REQUIRED_FILES, quantile_file
from src.api.service import Forecaster

BUNDLE_FILES = REQUIRED_FILES   # kept for backwards compatibility


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"No bucket in {uri!r}")
    return bucket, prefix.strip("/")


def _bundle_files(dest: Path) -> list[str]:
    """Required files, plus the quantile models that meta.json says exist."""
    meta = json.loads((dest / "meta.json").read_text())
    return [*REQUIRED_FILES, *(quantile_file(q) for q in meta.get("quantiles", []))]


def _complete(dest: Path) -> bool:
    if not all((dest / f).exists() for f in REQUIRED_FILES):
        return False
    return all((dest / f).exists() for f in _bundle_files(dest))


def download_bundle(s3_uri: str, dest: str | Path, client=None) -> Path:
    """Copy the bundle files from S3 into dest. Skips the download if dest already
    holds a complete bundle (a warm Lambda container keeps /tmp)."""
    dest = Path(dest)
    if _complete(dest):
        return dest
    if client is None:
        import boto3  # only needed on AWS; preinstalled in the Lambda base image
        client = boto3.client("s3")
    bucket, prefix = parse_s3_uri(s3_uri)
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        key = f"{prefix}/{name}" if prefix else name
        client.download_file(bucket, key, str(dest / name))

    for name in REQUIRED_FILES:
        fetch(name)
    for name in _bundle_files(dest)[len(REQUIRED_FILES):]:
        fetch(name)
    return dest


def load_forecaster(model_dir: str | Path) -> Forecaster | None:
    """Return a ready Forecaster, or None if no model is available (API answers 503)."""
    s3_uri = os.getenv("MODEL_S3_URI")
    if s3_uri:
        download_bundle(s3_uri, model_dir)
    return Forecaster.load(model_dir) if Path(model_dir).exists() else None
