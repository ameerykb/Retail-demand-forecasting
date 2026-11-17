"""Pydantic models: they define and VALIDATE what the API accepts and returns."""
import datetime as dt
from pydantic import BaseModel, Field, model_validator


class DayInput(BaseModel):
    date: dt.date
    is_open: int = Field(1, ge=0, le=1, description="1 if the store is open that day")
    promo: int = Field(0, ge=0, le=1, description="1 if a promotion runs that day")
    state_holiday: int = Field(0, ge=0, le=1)
    school_holiday: int = Field(0, ge=0, le=1)


class ForecastRequest(BaseModel):
    store: int = Field(ge=1, description="Store id")
    days: list[DayInput] = Field(min_length=1, max_length=42,
                                 description="Schedule for the days to forecast")

    @model_validator(mode="after")
    def dates_unique(self):
        dates = [d.date for d in self.days]
        if len(dates) != len(set(dates)):
            raise ValueError("Each date may appear only once.")
        return self


class DayForecast(BaseModel):
    date: dt.date
    predicted_sales: float


class ForecastResponse(BaseModel):
    store: int
    model_version: str
    last_history_date: dt.date
    forecasts: list[DayForecast]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    model_version: str
    algorithm: str
    horizon_days: int
    last_history_date: dt.date
    forecast_window_start: dt.date
    forecast_window_end: dt.date
    n_stores: int
    trained_on: str
    backtest_wape: float | None = None
    features: list[str]


class ErrorResponse(BaseModel):
    detail: str
