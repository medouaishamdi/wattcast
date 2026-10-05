"""Day-ahead features.

The forecast for day D is issued on D-1 around noon, as RTE does. Everything used for an hour of
day D must therefore be known by then: loads at least 36 hours old (we use 48 h and 7 days),
calendar facts, and the temperature forecast for D. Lags are computed on the hourly UTC index.
"""
from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from .config import TZ

FEATURES = [
    "hour", "dow", "doy_sin", "doy_cos", "is_holiday", "is_bridge", "near_holiday", "august", "xmas_break",
    "temp", "temp_smooth_24h", "temp_smooth_72h", "hdd", "cdd", "temp_day_min", "temp_day_max",
    "load_lag_48h", "load_lag_7d", "load_day_mean_lag_2d", "load_week_mean_lag",
    "temp_minus_lag_48h", "temp_minus_lag_7d", "years_since_2015",
]


@lru_cache(maxsize=1)
def _french_holidays(first: int = 2014, last: int = 2030) -> frozenset:
    return frozenset(holidays.France(years=range(first, last + 1)).keys())


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(TZ)
    day = pd.Series(local.date, index=index)
    hol = _french_holidays()
    is_hol = day.map(lambda d: d in hol).astype(int)
    prev_hol = day.map(lambda d: (d - pd.Timedelta(days=1)) in hol)
    next_hol = day.map(lambda d: (d + pd.Timedelta(days=1)) in hol)
    dow = local.dayofweek
    doy = local.dayofyear
    return pd.DataFrame({
        "hour": local.hour,
        "dow": dow,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_holiday": is_hol.values,
        # "pont": a Monday before a Tuesday holiday or a Friday after a Thursday one - many people take it off
        "is_bridge": (((dow == 0) & next_hol) | ((dow == 4) & prev_hol)).astype(int).values,
        "near_holiday": (prev_hol | next_hol).astype(int).values,
        "august": (local.month == 8).astype(int),
        "xmas_break": (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 2))).astype(int),
        "years_since_2015": (local.year - 2015) + doy / 365.25,
    }, index=index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: hourly frame with columns load and temp (UTC index). Returns FEATURES (+ load, rte columns kept)."""
    df = df.asfreq("1h")
    out = calendar(df.index)
    t = df["temp"]
    local_day = df.index.tz_convert(TZ).date
    out["temp"] = t
    # Buildings react to temperature with inertia: smoothed temperatures capture that.
    out["temp_smooth_24h"] = t.ewm(span=24).mean()
    out["temp_smooth_72h"] = t.ewm(span=72).mean()
    out["hdd"] = np.clip(15 - out["temp_smooth_24h"], 0, None)  # heating degrees (France heats a lot with electricity)
    out["cdd"] = np.clip(out["temp_smooth_24h"] - 20, 0, None)
    out["temp_day_min"] = t.groupby(local_day).transform("min")
    out["temp_day_max"] = t.groupby(local_day).transform("max")

    load = df["load"]
    out["load_lag_48h"] = load.shift(48)
    out["load_lag_7d"] = load.shift(168)
    daily_mean = load.groupby(local_day).transform("mean")
    out["load_day_mean_lag_2d"] = daily_mean.shift(48)
    out["load_week_mean_lag"] = load.shift(48).rolling(168, min_periods=120).mean()
    # How different is the weather now from when the lagged load was measured?
    out["temp_minus_lag_48h"] = out["temp_smooth_24h"] - out["temp_smooth_24h"].shift(48)
    out["temp_minus_lag_7d"] = out["temp_smooth_24h"] - out["temp_smooth_24h"].shift(168)

    for col in ("load", "rte_day_ahead"):
        if col in df:
            out[col] = df[col]
    return out

FEATURE_LABELS = {
    "hour": "Hour of the day", "dow": "Day of the week", "doy_sin": "Season (sine)", "doy_cos": "Season (cosine)",
    "is_holiday": "Public holiday", "is_bridge": "Bridge day (pont)", "near_holiday": "Day before/after a holiday",
    "august": "August", "xmas_break": "Christmas break", "temp": "Temperature",
    "temp_smooth_24h": "Temperature, 24 h smoothed", "temp_smooth_72h": "Temperature, 72 h smoothed",
    "hdd": "Heating degrees", "cdd": "Cooling degrees", "temp_day_min": "Day's minimum temperature",
    "temp_day_max": "Day's maximum temperature", "load_lag_48h": "Consumption 2 days earlier",
    "load_lag_7d": "Consumption 7 days earlier", "load_day_mean_lag_2d": "Average consumption 2 days earlier",
    "load_week_mean_lag": "Average consumption over the past week", "temp_minus_lag_48h": "Warmer/colder than 2 days earlier",
    "temp_minus_lag_7d": "Warmer/colder than 7 days earlier", "years_since_2015": "Long-term trend",
}
