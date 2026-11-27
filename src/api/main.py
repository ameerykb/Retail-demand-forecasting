"""FastAPI app. Run with:  uvicorn src.api.main:app --reload
Interactive docs: http://localhost:8000/docs"""
import datetime as dt
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from src.api.model_store import load_forecaster
from src.api.schemas import (DayForecast, ErrorResponse, ForecastRequest, ForecastResponse,
                             HealthResponse, ModelInfo)
from src.api.service import DateOutOfRange, Forecaster, StoreNotFound

ERRORS = {404: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
          503: {"model": ErrorResponse}}


def create_app(model_dir: str | None = None) -> FastAPI:
    model_dir = model_dir or os.getenv("MODEL_DIR", "models/forecaster")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # the model is loaded ONCE at startup, not on every request
        # (from disk, or from S3 first when MODEL_S3_URI is set)
        app.state.forecaster = load_forecaster(model_dir)
        yield

    app = FastAPI(title="Demand Forecast API", version="1.0.0", lifespan=lifespan,
                  description="Forecasts daily sales for Rossmann-style stores up to 42 days ahead.")

    def forecaster() -> Forecaster:
        if app.state.forecaster is None:
            raise HTTPException(503, "Model not loaded. Train one with: python train_final.py")
        return app.state.forecaster

    def run(store: int, days: list[dict]) -> ForecastResponse:
        fc = forecaster()
        try:
            out = fc.forecast(store, days)
        except StoreNotFound as e:
            raise HTTPException(404, str(e))
        except DateOutOfRange as e:
            raise HTTPException(422, str(e))
        return ForecastResponse(
            store=store, model_version=fc.meta["model_version"],
            last_history_date=fc.last_date.date(),
            forecasts=[DayForecast(date=r.date, predicted_sales=r.predicted_sales)
                       for r in out.itertuples()])

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    def health():
        loaded = app.state.forecaster is not None
        body = HealthResponse(status="ok" if loaded else "model_not_loaded", model_loaded=loaded)
        return JSONResponse(body.model_dump(), status_code=200 if loaded else 503)

    @app.get("/model-info", response_model=ModelInfo, responses=ERRORS)
    def model_info():
        fc = forecaster()
        lo, hi = fc.window
        m = fc.meta
        return ModelInfo(model_version=m["model_version"], algorithm=m["algorithm"],
                         horizon_days=fc.horizon, last_history_date=fc.last_date.date(),
                         forecast_window_start=lo.date(), forecast_window_end=hi.date(),
                         n_stores=len(fc.store_ids), trained_on=m["trained_on"],
                         backtest_wape=m.get("backtest_wape"), features=fc.feature_cols)

    @app.post("/forecast", response_model=ForecastResponse, responses=ERRORS)
    def forecast(req: ForecastRequest):
        days = [{"date": d.date, "open": d.is_open, "promo": d.promo,
                 "state_holiday": d.state_holiday, "school_holiday": d.school_holiday}
                for d in req.days]
        return run(req.store, days)

    @app.get("/forecast/{store}/default", response_model=ForecastResponse, responses=ERRORS)
    def forecast_default(store: int, n_days: int = Query(14, ge=1, le=42)):
        """Quick demo: forecast the next n_days assuming open Monday-Saturday, closed Sunday, no promo."""
        fc = forecaster()
        start = fc.last_date.date() + dt.timedelta(days=1)
        days = []
        for i in range(n_days):
            d = start + dt.timedelta(days=i)
            days.append({"date": d, "open": 0 if d.weekday() == 6 else 1, "promo": 0,
                         "state_holiday": 0, "school_holiday": 0})
        return run(store, days)

    return app


app = create_app()
