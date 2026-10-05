"""WattCast REST API.

Run from the project root:
    uvicorn api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import datetime as dt

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wattcast import __version__
from wattcast.service import ForecastService, ModelNotTrainedError, get_service

app = FastAPI(
    title="WattCast API",
    description="Day-ahead forecasts of French national electricity consumption (hourly, MW), with 80% prediction "
                "intervals, trained on RTE eCO2mix data and population-weighted temperatures.",
    version=__version__,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def service() -> ForecastService:
    try:
        return get_service()
    except ModelNotTrainedError as exc:
        raise HTTPException(503, str(exc)) from exc


class ForecastRequest(BaseModel):
    date: dt.date = Field(..., examples=["2026-07-01"])
    temperatures: list[float] | None = Field(
        None, description="Hourly temperature forecast (°C, population-weighted France) for each local hour of the day. "
                          "Omitted: observed temperature if known, otherwise the average of past years.",
        min_length=23, max_length=25)
    temp_offset: float = Field(0.0, ge=-15, le=15, description="Shift every temperature by this many °C (scenario).")


@app.get("/", tags=["meta"])
def root():
    return {"name": "WattCast API", "version": __version__, "docs": "/docs"}


@app.get("/health", tags=["meta"])
def health():
    try:
        s = get_service()
        return {"status": "healthy", "data_until": str(s.last_day), "can_forecast_until": str(s.last_forecastable_day)}
    except ModelNotTrainedError:
        return {"status": "degraded", "model_loaded": False}


@app.get("/model-info", tags=["meta"])
def model_info(svc: ForecastService = Depends(service)):
    m = svc.metadata
    b = m["backtest"]
    return {
        "version": m["version"], "trained_at": m["trained_at"], "data": m["data"],
        "backtest_mape": {m["model_names"][k]: round(v["mape"], 4) for k, v in b["overall"].items()},
        "backtest_by_year": b["by_year"], "interval_coverage_80": b["interval_coverage_80"],
        "thermosensitivity": m["thermosensitivity"], "temperature_forecast_error": m["temperature_forecast_error"],
    }


@app.post("/forecast", tags=["forecast"])
def forecast(req: ForecastRequest, svc: ForecastService = Depends(service)):
    """Forecast the 24 hours of a day (23/25 on clock-change days). Days up to two days after the end of the
    data can be forecast, since the forecast is made the day before with consumption known up to two days before."""
    try:
        return svc.forecast(req.date, req.temperatures, req.temp_offset)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/forecast/{day}", tags=["forecast"])
def forecast_get(day: dt.date, temp_offset: float = Query(0.0, ge=-15, le=15), svc: ForecastService = Depends(service)):
    """Same as POST /forecast with default temperatures."""
    try:
        return svc.forecast(day, None, temp_offset)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/backtest/day/{day}", tags=["backtest"])
def backtest_day(day: dt.date, svc: ForecastService = Depends(service)):
    """The out-of-sample forecast made for a past day by a model trained only on earlier years, next to the
    actual consumption and RTE's own forecast."""
    try:
        return svc.backtest_day(day)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
