# Changelog

## 1.0.0 (2026-11-27)

### Added
- Calibrated 80% prediction intervals: quantile LightGBM (p10/p90) widened by conformal calibration (CQR); returned by the API as `lower` / `upper`.
- Hyperparameter tuning with Optuna on a period *before* the reported backtest (`scripts/tune_lgbm.py`).
- Explainability: TreeSHAP feature impact chart (`scripts/explain.py`).
- Serverless deployment on AWS: Lambda container image, API Gateway, S3 (versioned), ECR, Terraform; JSON request logs in CloudWatch.
- `/model-info` lists the store ids and the interval; `/` redirects to the docs.
- Streamlit demo ready for Streamlit Community Cloud (store picker, cold-start retries, interval band).
- Dependabot, pre-commit, stricter Ruff rules, MIT license.

### Changed
- Scripts moved to `scripts/` and run as modules (`python -m scripts.train_final`); shared settings in `src/config.py`.
- One backtest module (`src/training/backtest.py`) instead of three.
- All dependencies pinned; CI on Python 3.12 (same as Lambda).
- The API builds features for the requested store only (faster responses).

## 0.1.0
- Leak-free features, rolling backtest, baselines, LightGBM vs PyTorch MLP, MLflow, FastAPI, Docker, Streamlit demo, CI.
