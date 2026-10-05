"""Tests for data, features (no look-ahead), models, service and API."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wattcast.config import TZ  # noqa: E402
from wattcast.data import load_hourly  # noqa: E402
from wattcast.features import FEATURE_LABELS, FEATURES, build_features, calendar  # noqa: E402
from wattcast.models import metrics  # noqa: E402
from wattcast.service import day_hours, get_service  # noqa: E402


@pytest.fixture(scope="session")
def hourly():
    return load_hourly()


@pytest.fixture(scope="session")
def svc():
    return get_service()


# -------------------------------------------------------------------- data
def test_hourly_table(hourly):
    assert len(hourly) > 100_000
    assert {"load", "rte_day_ahead", "temp"} <= set(hourly.columns)
    assert hourly["load"].between(25_000, 100_000).all()
    assert hourly["temp"].between(-15, 45).all()
    assert hourly.index.is_monotonic_increasing and not hourly.index.duplicated().any()


# ---------------------------------------------------------------- features
def test_no_look_ahead(hourly):
    """Changing consumption after D-2 must not change any feature of day D (forecast made on D-1)."""
    day = pd.Timestamp("2025-02-12", tz=TZ)
    window = hourly.loc[day - pd.Timedelta(days=12): day + pd.Timedelta(days=1)].copy()
    base = build_features(window)
    tampered = window.copy()
    recent = tampered.index >= (day - pd.Timedelta(days=1)).tz_convert("UTC")  # all of D-1 and D
    tampered.loc[recent, "load"] *= 1.5
    after = build_features(tampered)
    target = base.index[(base.index >= day.tz_convert("UTC")) & (base.index < (day + pd.Timedelta(days=1)).tz_convert("UTC"))]
    pd.testing.assert_frame_equal(base.loc[target, FEATURES], after.loc[target, FEATURES])


def test_calendar_holidays_and_bridges():
    idx = pd.date_range("2024-05-08 12:00", periods=3, freq="1D", tz=TZ).tz_convert("UTC")
    cal = calendar(idx)  # 8 May (holiday, Wed), 9 May (Ascension, Thu), 10 May (Fri bridge)
    assert cal["is_holiday"].tolist() == [1, 1, 0]
    assert cal["is_bridge"].tolist() == [0, 0, 1]
    assert set(FEATURES) <= set(FEATURE_LABELS)


def test_dst_days_have_23_and_25_hours():
    assert len(day_hours(date(2025, 3, 30))) == 23
    assert len(day_hours(date(2025, 10, 26))) == 25
    assert len(day_hours(date(2025, 6, 1))) == 24


def test_metrics():
    m = metrics(np.array([100, 200]), np.array([110, 180]))
    assert m["mape"] == pytest.approx(0.1)
    assert m["mae_mw"] == pytest.approx(15)
    assert m["bias_mw"] == pytest.approx(-5)


# ----------------------------------------------------------------- results
def test_backtest_results_hold(svc):
    b = svc.metadata["backtest"]
    o = b["overall"]
    assert o["lightgbm"]["mape"] < 0.025
    assert o["lightgbm"]["mape"] < o["linear"]["mape"] < o["naive_7d"]["mape"]
    assert 0.75 <= b["interval_coverage_80"] <= 0.85
    assert {r["year"] for r in b["by_year"]} >= {2019, 2025}


def test_heating_slope_is_physical(svc):
    for row in svc.metadata["thermosensitivity"]:
        assert 1500 < row["mw_per_degree"] < 3000  # RTE quotes about 2,400 MW per deg C


# ----------------------------------------------------------------- service
def test_forecast_inside_data(svc):
    f = svc.forecast(date(2025, 1, 15))
    assert f["hours"] == 24 and f["temperature_source"] == "observed"
    h = pd.DataFrame(f["hourly"])
    assert (h["low_mw"] <= h["forecast_mw"]).all() and (h["forecast_mw"] <= h["high_mw"]).all()
    assert f["mape"] < 0.06
    assert f["in_training_data"] and "note" in f


def test_forecast_beyond_data_uses_climatology(svc):
    f = svc.forecast(svc.last_forecastable_day)
    assert f["temperature_source"].startswith("climatology")
    assert all(r["actual_mw"] is None for r in f["hourly"])
    with pytest.raises(ValueError):
        svc.forecast(date(2030, 1, 1))


def test_colder_day_means_more_demand_in_winter(svc):
    base = svc.forecast(date(2026, 1, 20))
    cold = svc.forecast(date(2026, 1, 20), temp_offset=-4)
    assert cold["daily_energy_gwh"] > base["daily_energy_gwh"]


def test_custom_temperatures_and_dst(svc):
    f = svc.forecast(date(2025, 3, 30), temperatures=[10.0] * 23)
    assert f["hours"] == 23 and f["temperature_source"] == "provided"
    with pytest.raises(ValueError):
        svc.forecast(date(2025, 3, 30), temperatures=[10.0] * 24)


def test_backtest_day_is_out_of_sample(svc):
    r = svc.backtest_day(date(2024, 1, 10))
    assert r["trained_on"] == "2015 to 2023"
    assert 0 < r["mape"] < 0.1
    with pytest.raises(ValueError):
        svc.backtest_day(date(2016, 1, 1))


# --------------------------------------------------------------------- API
@pytest.fixture(scope="module")
def client(svc):
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


def test_api(client, svc):
    assert client.get("/health").json()["status"] == "healthy"
    info = client.get("/model-info").json()
    assert "LightGBM (WattCast)" in info["backtest_mape"]
    r = client.post("/forecast", json={"date": str(svc.last_forecastable_day)})
    assert r.status_code == 200 and len(r.json()["hourly"]) == r.json()["hours"]
    assert client.get("/forecast/2025-01-15?temp_offset=-2").status_code == 200
    assert client.post("/forecast", json={"date": "2030-01-01"}).status_code == 422
    assert client.post("/forecast", json={"date": "2025-01-15", "temperatures": [1.0] * 5}).status_code == 422
    assert client.get("/backtest/day/2024-01-10").status_code == 200
    assert client.get("/backtest/day/2016-01-01").status_code == 404
