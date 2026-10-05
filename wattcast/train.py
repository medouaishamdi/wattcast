"""Backtest every model year by year, run the experiments, and train the final model.

    python -m wattcast.train

Backtest = expanding window: to forecast year Y, models are trained on 2015 .. Y-1 only.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .config import (BACKTEST_PATH, FIGURES_DIR, METADATA_PATH, MODEL_PATH, MODELS_DIR, RANDOM_STATE, TEST_YEARS, TZ)
from .data import load_hourly
from .features import FEATURES, build_features
from .models import Forecaster, make_linear, metrics

MODEL_NAMES = {
    "naive_7d": "Same hour last week",
    "linear": "Linear regression",
    "lightgbm": "LightGBM (WattCast)",
    "rte": "RTE's published day-ahead forecast",
    "rte_debiased": "RTE's forecast, bias-corrected",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def year_of(index):
    return index.tz_convert(TZ).year


def dataset(df: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(df)
    return feats.dropna(subset=FEATURES + ["load"])


def rte_debiased(feats: pd.DataFrame) -> pd.Series:
    """RTE's published forecast minus its own average error over the previous 28 days (known at forecast time:
    errors up to 2 days old). Since 2023 the published forecast sits ~1.1 GW below published consumption, even
    for the same-day forecast, which suggests a perimeter difference between the two series, not bad forecasting."""
    err = (feats["rte_day_ahead"] - feats["load"]).asfreq("1h")
    bias = err.shift(48).rolling(28 * 24, min_periods=7 * 24).mean()
    return (feats["rte_day_ahead"] - bias.reindex(feats.index)).rename("rte_debiased")


def backtest(feats: pd.DataFrame) -> pd.DataFrame:
    years = year_of(feats.index)
    rte_fixed = rte_debiased(feats)
    parts = []
    for y in TEST_YEARS:
        train, test = feats[years < y], feats[years == y]
        if test.empty:
            continue
        t0 = time.time()
        lin = make_linear().fit(train, train["load"])
        fc = Forecaster().fit(train, train["load"])
        p = fc.predict(test)
        parts.append(pd.DataFrame({
            "load": test["load"], "rte": test["rte_day_ahead"], "rte_debiased": rte_fixed[test.index], "naive_7d": test["load_lag_7d"],
            "linear": lin.predict(test), "lightgbm": p["forecast"], "low": p["low"], "high": p["high"],
            "temp": test["temp"], "year": y,
        }, index=test.index))
        log(f"backtest {y}: {len(train):,} training hours, {time.time() - t0:.0f}s, "
            f"MAPE lightgbm {metrics(test['load'], p['forecast'])['mape']:.2%} vs RTE {metrics(test['load'], test['rte_day_ahead'])['mape']:.2%}")
    return pd.concat(parts)


def summarize(bt: pd.DataFrame) -> dict:
    out = {"overall": {}, "by_year": []}
    for key in MODEL_NAMES:
        out["overall"][key] = metrics(bt["load"], bt[key])
    for y, g in bt.groupby("year"):
        row = {"year": int(y), "hours": len(g)}
        for key in MODEL_NAMES:
            row[key] = metrics(g["load"], g[key])["mape"]
        row["interval_coverage"] = float(((g["load"] >= g["low"]) & (g["load"] <= g["high"])).mean())
        out["by_year"].append(row)
    out["interval_coverage_80"] = float(((bt["load"] >= bt["low"]) & (bt["load"] <= bt["high"])).mean())
    local = bt.index.tz_convert(TZ)
    out["mape_by_hour"] = {int(h): metrics(g["load"], g["lightgbm"])["mape"] for h, g in bt.groupby(local.hour)}
    daily = bt.assign(day=local.date).groupby("day").apply(
        lambda g: pd.Series({"ape": float(np.mean(np.abs(g.lightgbm - g.load) / g.load)),
                             "rte_ape": float(np.mean(np.abs(g.rte - g.load) / g.load)),
                             "temp": float(g.temp.mean())}), include_groups=False)
    worst = daily.sort_values("ape", ascending=False).head(10)
    out["worst_days"] = [{"day": str(d), "mape": r.ape, "rte_mape": r.rte_ape, "temp": r.temp} for d, r in worst.iterrows()]
    return out


def frozen_model_experiment(feats: pd.DataFrame) -> list[dict]:
    """Train once on 2015-2021 and never retrain: what happens when consumption habits change (2022 energy crisis)?"""
    years = year_of(feats.index)
    train = feats[years <= 2021]
    fc = Forecaster().fit(train, train["load"])
    rows = []
    for y in range(2022, max(TEST_YEARS) + 1):
        test = feats[years == y]
        if test.empty:
            continue
        m = metrics(test["load"], fc.predict(test)["forecast"])
        rows.append({"year": y, "mape_frozen": m["mape"], "bias_mw_frozen": m["bias_mw"]})
    return rows


def temperature_forecast_error_experiment(df: pd.DataFrame, fc: Forecaster, year: int, sigma: float = 1.5) -> dict:
    """The backtest uses observed temperatures (a perfect weather forecast). Real day-ahead temperature
    forecasts are off by roughly 1-2 deg C; add that much error and measure the cost."""
    rng = np.random.default_rng(RANDOM_STATE)
    noisy = df.copy()
    local_day = pd.Series(noisy.index.tz_convert(TZ).date, index=noisy.index)
    day_error = local_day.map(dict(zip(local_day.unique(), rng.normal(0, sigma, local_day.nunique()))))
    noisy["temp"] = noisy["temp"] + day_error.values + rng.normal(0, 0.5, len(noisy))
    clean_f, noisy_f = dataset(df), dataset(noisy)
    sel = year_of(clean_f.index) == year
    clean_m = metrics(clean_f.loc[sel, "load"], fc.predict(clean_f[sel])["forecast"])
    sel_n = year_of(noisy_f.index) == year
    noisy_m = metrics(noisy_f.loc[sel_n, "load"], fc.predict(noisy_f[sel_n])["forecast"])
    return {"year": year, "temp_error_sigma_c": sigma, "mape_observed_temp": clean_m["mape"], "mape_noisy_temp": noisy_m["mape"]}


def daily_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    """Daily means for ordinary working days (no weekends, public holidays or Christmas break)."""
    from .features import _french_holidays

    local = df.index.tz_convert(TZ)
    daily = df.groupby(local.date)[["load", "temp"]].mean()
    idx = pd.to_datetime(daily.index)
    hol = _french_holidays()
    ordinary = ((idx.dayofweek < 5) & ~pd.Index(daily.index).isin(hol)
                & ~((idx.month == 12) & (idx.day >= 20)) & ~((idx.month == 1) & (idx.day <= 3)))
    return daily[ordinary].assign(year=idx[ordinary].year)


def thermosensitivity(df: pd.DataFrame, below: float = 14.0) -> list[dict]:
    """Slope of the heating curve: extra MW per deg C colder, on working days colder than `below` deg C.
    Measured on the data (not the model): a day-ahead model anchors on recent real consumption, so asking it
    "what if tomorrow were 1 deg C colder" understates the effect of a lasting cold spell."""
    daily = daily_weekdays(df)
    rows = []
    for first, last in [(2015, 2019), (2023, 2026)]:
        d = daily[(daily.year >= first) & (daily.year <= last) & (daily.temp < below)]
        slope, intercept = np.polyfit(d["temp"], d["load"], 1)
        rows.append({"period": f"{first}-{last}", "mw_per_degree": float(-slope), "intercept": float(intercept), "days": len(d)})
    return rows


def make_figures(bt: pd.DataFrame, summary: dict, frozen: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .style import BASELINE, SERIES, TEXT_SECONDARY, apply_matplotlib_style

    apply_matplotlib_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. MAPE by year and model
    by_year = pd.DataFrame(summary["by_year"]).set_index("year")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    keys = ["naive_7d", "linear", "lightgbm", "rte", "rte_debiased"]
    w = 0.16
    for i, k in enumerate(keys):
        ax.bar(np.arange(len(by_year)) + (i - 2) * w, by_year[k] * 100, w, label=MODEL_NAMES[k], color=SERIES[i])
    ax.set_xticks(np.arange(len(by_year)), [f"{y}{'*' if y == 2026 else ''}" for y in by_year.index])
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Day-ahead forecast error by year (lower is better)", loc="left")
    ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=8.5)
    ax.set_ylim(0, by_year[keys].values.max() * 100 * 1.35)
    ax.text(1, -0.14, "* 2026 = January to June. Each year is forecast by a model trained only on the years before it.",
            transform=ax.transAxes, ha="right", fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mape_by_year.png", dpi=150)
    plt.close(fig)

    # 2. A winter week: actual vs forecast with interval vs RTE
    local = bt.index.tz_convert(TZ)
    week = bt[(local >= pd.Timestamp("2026-01-12", tz=TZ)) & (local < pd.Timestamp("2026-01-19", tz=TZ))]
    if week.empty:
        week = bt.iloc[-168:]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = week.index.tz_convert(TZ)
    ax.fill_between(x, week["low"] / 1000, week["high"] / 1000, color=SERIES[0], alpha=0.15, lw=0, label="WattCast 80% interval")
    ax.plot(x, week["load"] / 1000, color="#0b0b0b", lw=1.8, label="Actual consumption")
    ax.plot(x, week["lightgbm"] / 1000, color=SERIES[0], lw=1.4, label="WattCast forecast")
    ax.plot(x, week["rte"] / 1000, color=SERIES[3], lw=1.2, ls="--", label="RTE day-ahead forecast")
    ax.set_ylabel("GW")
    ax.set_title("A winter week (12-18 January 2026), forecast the day before", loc="left")
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "winter_week.png", dpi=150)
    plt.close(fig)

    # 3. Frozen vs retrained after 2022
    fr = pd.DataFrame(frozen).set_index("year")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(fr.index, fr["mape_frozen"] * 100, "o-", color=SERIES[1], label="Trained once on 2015-2021, never updated")
    ax.plot(by_year.loc[fr.index].index, by_year.loc[fr.index, "lightgbm"] * 100, "o-", color=SERIES[0], label="Retrained every year")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(fr.index)
    ax.set_ylim(0, None)
    ax.set_title("After the 2022 energy crisis, a model never retrained falls behind", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "frozen_vs_retrained.png", dpi=150)
    plt.close(fig)

    # 4. The heating curve: working days, two periods, fitted slopes below 14 deg C
    daily = daily_weekdays(load_hourly())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for color, (first, last) in zip((SERIES[0], SERIES[1]), [(2015, 2019), (2023, 2026)]):
        d = daily[(daily.year >= first) & (daily.year <= last)]
        ax.scatter(d["temp"], d["load"] / 1000, s=6, alpha=0.35, color=color, lw=0)
        cold = d[d.temp < 14]
        slope, intercept = np.polyfit(cold["temp"], cold["load"], 1)
        xs = np.array([cold.temp.min(), 14])
        ax.plot(xs, (slope * xs + intercept) / 1000, color=color, lw=2.2,
                label=f"{first}-{last}: +{-slope:,.0f} MW per °C colder")
    ax.set_xlabel("Daily mean temperature, population-weighted (°C)")
    ax.set_ylabel("Daily mean consumption (GW)")
    ax.set_title("France heats with electricity: the heating curve (working days)", loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heating_curve.png", dpi=150)
    plt.close(fig)


def main():
    t_start = time.time()
    df = load_hourly()
    feats = dataset(df)
    log(f"{len(feats):,} hours with complete features, {feats.index.min()} -> {feats.index.max()}")

    bt = backtest(feats)
    summary = summarize(bt)
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    bt.astype({c: "float32" for c in bt.columns if c != "year"}).to_parquet(BACKTEST_PATH)

    log("frozen-model experiment")
    frozen = frozen_model_experiment(feats)

    log("training the final model on everything")
    final = Forecaster().fit(feats, feats["load"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)

    # Experiments that need a model that hasn't seen the evaluated year: reuse the backtest setup for 2025.
    log("temperature-error and thermosensitivity experiments")
    years = year_of(feats.index)
    fc_2025 = Forecaster().fit(feats[years < 2025], feats.loc[years < 2025, "load"])
    temp_exp = temperature_forecast_error_experiment(df, fc_2025, 2025)
    thermo = thermosensitivity(df)

    importance = pd.Series(final.point.booster_.feature_importance("gain"), index=FEATURES)
    importance = (importance / importance.sum()).sort_values(ascending=False)

    meta = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_until": final.trained_until,
        "data": {"first_hour": str(df.index.min()), "last_hour": str(df.index.max()), "hours": len(df)},
        "model_names": MODEL_NAMES,
        "backtest": summary,
        "frozen_model": frozen,
        "temperature_forecast_error": temp_exp,
        "thermosensitivity": thermo,
        "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    make_figures(bt, summary, frozen)

    o = summary["overall"]
    log("backtest 2019 - mid 2026, MAPE: " + ", ".join(f"{MODEL_NAMES[k]} {o[k]['mape']:.2%}" for k in MODEL_NAMES))
    log(f"80% interval coverage: {summary['interval_coverage_80']:.1%}")
    log(f"frozen model: {frozen}")
    log(f"temperature error: {temp_exp}")
    log(f"thermosensitivity: {thermo}")
    log(f"done in {time.time() - t_start:.0f}s")


def figures_only():
    """Redraw the figures from saved backtest results (no retraining)."""
    meta = json.loads(METADATA_PATH.read_text())
    make_figures(pd.read_parquet(BACKTEST_PATH), meta["backtest"], meta["frozen_model"])


if __name__ == "__main__":
    import sys

    figures_only() if "--figures-only" in sys.argv else main()
