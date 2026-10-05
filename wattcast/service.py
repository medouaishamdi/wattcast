"""ForecastService: everything the API and the dashboard need."""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import cached_property, lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import BACKTEST_PATH, METADATA_PATH, MODEL_PATH, TZ
from .data import load_hourly
from .features import FEATURES, build_features


class ModelNotTrainedError(RuntimeError):
    pass


def day_hours(day: date) -> pd.DatetimeIndex:
    """The hours of a local calendar day, in UTC (23 or 25 of them on daylight-saving days)."""
    start = pd.Timestamp(day, tz=TZ)
    end = pd.Timestamp(day + timedelta(days=1), tz=TZ)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


class ForecastService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise ModelNotTrainedError("No trained model. Run `python -m wattcast.train` first.")
        self.model = joblib.load(MODEL_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text())
        self.history = load_hourly()

    @cached_property
    def backtest(self) -> pd.DataFrame:
        return pd.read_parquet(BACKTEST_PATH)

    @property
    def last_day(self) -> date:
        return self.history.index.max().tz_convert(TZ).date()

    @property
    def first_forecastable_day(self) -> date:
        return self.history.index.min().tz_convert(TZ).date() + timedelta(days=10)

    @property
    def last_forecastable_day(self) -> date:
        # Day D needs actual consumption up to D-2 (the forecast is made on D-1 at noon).
        return self.last_day + timedelta(days=2)

    @cached_property
    def climatology(self) -> pd.Series:
        """Average temperature for each (month, day, hour) over all years: the fallback 'forecast'."""
        local = self.history.index.tz_convert(TZ)
        return self.history["temp"].groupby([local.month, local.day, local.hour]).mean()

    def default_temperatures(self, day: date) -> tuple[np.ndarray, str]:
        hours = day_hours(day)
        known = self.history["temp"].reindex(hours)
        if known.notna().sum() >= len(hours) - 2:  # the doubled autumn hour can be missing: fill it
            return known.interpolate().bfill().ffill().to_numpy(), "observed"
        local = hours.tz_convert(TZ)
        clim = np.array([self.climatology.get((m, d, h), np.nan) for m, d, h in zip(local.month, local.day, local.hour)])
        return pd.Series(clim).interpolate().bfill().ffill().to_numpy(), "climatology (average of past years)"

    # ------------------------------------------------------------ forecasting
    def forecast(self, day: date, temperatures: list[float] | None = None, temp_offset: float = 0.0) -> dict:
        if not (self.first_forecastable_day <= day <= self.last_forecastable_day):
            raise ValueError(f"Dates from {self.first_forecastable_day} to {self.last_forecastable_day} can be forecast "
                             f"(the model needs the actual consumption up to two days before).")
        hours = day_hours(day)
        if temperatures is not None:
            if len(temperatures) != len(hours):
                raise ValueError(f"{day} has {len(hours)} hours in French local time: give {len(hours)} temperatures.")
            temps, source = np.asarray(temperatures, float), "provided"
        else:
            temps, source = self.default_temperatures(day)
        temps = temps + temp_offset

        window_start = hours[0] - pd.Timedelta(days=10)
        hist = self.history.loc[window_start:hours[0] - pd.Timedelta(hours=1)].copy()
        target = pd.DataFrame({"temp": temps}, index=hours)
        frame = pd.concat([hist[["load", "temp", "rte_day_ahead"]], target])
        frame = frame[~frame.index.duplicated(keep="last")]
        feats = build_features(frame).loc[hours]
        if feats[FEATURES].isna().any().any():
            raise ValueError(f"Missing history to forecast {day}.")
        pred = self.model.predict(feats)

        actual = self.history.reindex(hours)
        local = hours.tz_convert(TZ)
        table = pd.DataFrame({
            "time": local.strftime("%Y-%m-%d %H:%M"),
            "hour": local.hour,
            "temperature": np.round(temps, 1),
            "forecast_mw": pred["forecast"].round(0).to_numpy(),
            "low_mw": pred["low"].round(0).to_numpy(),
            "high_mw": pred["high"].round(0).to_numpy(),
            "actual_mw": actual["load"].round(0).to_numpy(),
            "rte_forecast_mw": actual["rte_day_ahead"].round(0).to_numpy(),
        })
        in_training = pd.Timestamp(hours[-1]) <= pd.Timestamp(self.model.trained_until)
        out = {
            "date": str(day),
            "hours": len(hours),
            "temperature_source": source + (f", shifted by {temp_offset:+.1f} °C" if temp_offset else ""),
            "in_training_data": bool(in_training),
            "daily_energy_gwh": round(float(pred["forecast"].sum()) / 1000, 1),
            "peak_mw": int(pred["forecast"].max()),
            "peak_hour": int(local[int(np.argmax(pred["forecast"].to_numpy()))].hour),
            "hourly": table.replace({np.nan: None}).to_dict(orient="records"),
        }
        if table["actual_mw"].notna().all():
            out["mape"] = round(float(np.mean(np.abs(table.forecast_mw - table.actual_mw) / table.actual_mw)), 4)
        if in_training:
            out["note"] = ("This day was part of the final model's training data, so its error is optimistic. "
                           "Use /backtest/day for out-of-sample forecasts.")
        return out

    def backtest_day(self, day: date) -> dict:
        """The out-of-sample forecast made for `day` during the backtest (by a model trained on earlier years only)."""
        bt = self.backtest
        rows = bt.loc[bt.index.tz_convert(TZ).date == day]
        if rows.empty:
            first, last = bt.index.min().tz_convert(TZ).date(), bt.index.max().tz_convert(TZ).date()
            raise ValueError(f"The backtest covers {first} to {last}.")
        local = rows.index.tz_convert(TZ)
        table = pd.DataFrame({
            "time": local.strftime("%Y-%m-%d %H:%M"), "hour": local.hour, "temperature": rows["temp"].round(1).to_numpy(),
            "forecast_mw": rows["lightgbm"].round(0).to_numpy(), "low_mw": rows["low"].round(0).to_numpy(),
            "high_mw": rows["high"].round(0).to_numpy(), "actual_mw": rows["load"].round(0).to_numpy(),
            "rte_forecast_mw": rows["rte"].round(0).to_numpy(),
        })
        ape = lambda col: float(np.mean(np.abs(table[col] - table.actual_mw) / table.actual_mw))  # noqa: E731
        return {
            "date": str(day), "trained_on": f"2015 to {int(rows['year'].iloc[0]) - 1}",
            "mape": round(ape("forecast_mw"), 4), "rte_mape": round(ape("rte_forecast_mw"), 4),
            "hourly": table.replace({np.nan: None}).to_dict(orient="records"),
        }


@lru_cache(maxsize=1)
def get_service() -> ForecastService:
    return ForecastService()
