"""Raw RTE eCO2mix + Open-Meteo files -> one clean hourly table.

Raw files (see README "Data") go in data/raw/:
  eco2mix-national-cons-def.csv     RTE national consumption, 15/30-min, with RTE's own forecasts
  weather_*.csv                     Open-Meteo hourly 2 m temperature for the cities in config.CITIES
The processed table is committed to the repo, so the raw files are only needed to rebuild it.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .config import CITIES, FIRST_YEAR, PROCESSED_PATH, RAW_DIR, TZ


def load_eco2mix(path=None) -> pd.DataFrame:
    path = path or RAW_DIR / "eco2mix-national-cons-def.csv"
    e = pd.read_csv(path, sep=";", usecols=["date_heure", "consommation", "prevision_j1", "prevision_j"])
    e["time"] = pd.to_datetime(e["date_heure"], utc=True)
    e = e.drop(columns="date_heure").drop_duplicates("time").set_index("time").sort_index()
    # Consumption is published every 30 min, forecasts every 15 min: average both to hourly MW.
    hourly = e.resample("1h").mean()
    hourly.columns = ["load", "rte_day_ahead", "rte_intraday"]
    return hourly


def load_weather(pattern=None) -> pd.Series:
    """Population-weighted mean temperature (deg C), hourly, UTC index."""
    weights = np.array([w for _, _, w in CITIES.values()])
    frames = []
    for path in sorted(glob.glob(str(pattern or RAW_DIR / "weather_*.csv"))):
        raw = pd.read_csv(path, skiprows=len(CITIES) + 2)  # skip the location header block
        raw.columns = ["location", "time", "temp"]
        frames.append(raw)
    w = pd.concat(frames).drop_duplicates(["location", "time"])
    w = w.pivot(index="time", columns="location", values="temp").sort_index()
    if w.shape[1] != len(CITIES):
        raise ValueError(f"Expected {len(CITIES)} locations in the weather files, found {w.shape[1]}")
    local = pd.to_datetime(w.index)
    # Open-Meteo gives local wall-clock times: the autumn DST hour appears twice, the spring one never.
    w.index = local.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    w = w[w.index.notna()]
    w.index = w.index.tz_convert("UTC")
    temp = pd.Series(w.values @ weights / weights.sum(), index=w.index, name="temp")
    return temp[~temp.index.duplicated()].resample("1h").mean().interpolate(limit=3)


def build_hourly() -> pd.DataFrame:
    df = load_eco2mix().join(load_weather(), how="inner")
    df = df[df.index.tz_convert(TZ).year >= FIRST_YEAR]
    df = df.dropna(subset=["load"])
    df["temp"] = df["temp"].interpolate(limit=6)
    return df.dropna(subset=["temp"])


def load_hourly(rebuild: bool = False) -> pd.DataFrame:
    if rebuild or not PROCESSED_PATH.exists():
        df = build_hourly()
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.astype("float32").to_parquet(PROCESSED_PATH)
    return pd.read_parquet(PROCESSED_PATH)


if __name__ == "__main__":
    d = load_hourly(rebuild=True)
    print(d.describe().round(1))
    print(d.index.min(), "->", d.index.max(), len(d), "hours")
