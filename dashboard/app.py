"""WattCast dashboard.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wattcast.config import FIGURES_DIR, TZ  # noqa: E402
from wattcast.features import FEATURE_LABELS  # noqa: E402
from wattcast.service import ModelNotTrainedError, get_service  # noqa: E402

st.set_page_config(page_title="WattCast", page_icon="⚡", layout="wide")

C_ACTUAL, C_OURS, C_RTE = "#8a8a8a", "#2a78d6", "#eda100"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1300px;}
      div[data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.18);
          border-radius: 10px; padding: 12px 14px;}
      .callout {border: 1px solid rgba(127,127,127,0.3); border-radius: 10px; padding: 12px 16px; margin: 8px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", title_font=dict(size=15), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"))
    fig.update_xaxes(gridcolor="rgba(127,127,127,0.2)")
    fig.update_yaxes(gridcolor="rgba(127,127,127,0.2)", zerolinecolor="rgba(127,127,127,0.3)")
    return fig


def day_chart(hourly: list[dict], title: str, show_rte: bool = True) -> go.Figure:
    h = pd.DataFrame(hourly)
    x = h["time"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=h["high_mw"] / 1000, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=h["low_mw"] / 1000, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.15)", name="80% interval", hoverinfo="skip"))
    if h["actual_mw"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=h["actual_mw"] / 1000, name="Actual", line=dict(color=C_ACTUAL, width=3)))
    fig.add_trace(go.Scatter(x=x, y=h["forecast_mw"] / 1000, name="WattCast", line=dict(color=C_OURS, width=2.5)))
    if show_rte and h["rte_forecast_mw"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=h["rte_forecast_mw"] / 1000, name="RTE forecast",
                                 line=dict(color=C_RTE, width=2, dash="dash")))
    fig.update_layout(title=title, yaxis_title="GW")
    return style_fig(fig)


@st.cache_resource(show_spinner="Loading model and 11 years of data...")
def svc():
    return get_service()


try:
    service = svc()
except ModelNotTrainedError as e:
    st.error(str(e))
    st.stop()
meta = service.metadata
bt = meta["backtest"]
names = meta["model_names"]

with st.sidebar:
    st.markdown("## ⚡ WattCast")
    st.caption("Day-ahead forecasts of French electricity demand")
    st.divider()
    st.metric("Backtest error (MAPE)", f"{bt['overall']['lightgbm']['mape']:.2%}")
    st.metric("Average error", f"{bt['overall']['lightgbm']['mae_mw']:,.0f} MW")
    st.metric("80% interval actually covers", f"{bt['interval_coverage_80']:.0%}")
    st.caption(f"Backtest: every hour from 2019 to June 2026. Data: RTE eCO2mix + Open-Meteo, January 2015 to {service.last_day:%B %Y}.")

tab_past, tab_next, tab_bt, tab_drivers, tab_how = st.tabs(
    ["Replay a past day", "Forecast the next days", "Backtest results", "What drives demand", "How it works"])

# --------------------------------------------------------------- past days
with tab_past:
    st.header("How good was the forecast on a given day?")
    st.caption("Each day below was forecast by a model trained only on the years before it, exactly as if it had "
               "been running live. Pick a day, or one of the interesting ones.")
    presets = {
        "Pick a day...": None,
        "Cold snap: 12 January 2026": date(2026, 1, 12),
        "COVID lockdown starts: 17 March 2020": date(2020, 3, 17),
        "Heatwave: 25 July 2019 (Paris record 42.6 °C)": date(2019, 7, 25),
        "Christmas Day 2024": date(2024, 12, 25),
        "Bridge day (pont): Friday 10 May 2024": date(2024, 5, 10),
        "Clock change: 30 March 2025 (23 hours)": date(2025, 3, 30),
    }
    c1, c2 = st.columns([2, 1])
    preset = c1.selectbox("Interesting days", list(presets))
    first_bt = pd.Timestamp(service.backtest.index.min()).tz_convert(TZ).date() + timedelta(days=1)
    default = presets[preset] or date(2026, 1, 12)
    chosen = c2.date_input("Or any day", value=default, min_value=first_bt, max_value=service.last_day,
                           key=f"day_{preset}")
    res = service.backtest_day(chosen)
    m = st.columns(4)
    m[0].metric("WattCast error (MAPE)", f"{res['mape']:.2%}")
    m[1].metric("RTE's published forecast", f"{res['rte_mape']:.2%}",
                help="RTE's published forecast runs about 1.1 GW below published consumption since 2023 "
                     "(a definition difference), which inflates its error here.")
    h = pd.DataFrame(res["hourly"])
    m[2].metric("Peak consumption", f"{h['actual_mw'].max() / 1000:.1f} GW")
    m[3].metric("Mean temperature", f"{h['temperature'].mean():.1f} °C")
    st.plotly_chart(day_chart(res["hourly"], f"{chosen:%A %d %B %Y} · model trained on {res['trained_on']}"),
                    use_container_width=True)
    with st.expander("Hourly table"):
        st.dataframe(h.drop(columns="hour"), hide_index=True, use_container_width=True)

# --------------------------------------------------------------- next days
with tab_next:
    st.header("Forecast beyond the data")
    st.markdown(f"The data ends on **{service.last_day:%d %B %Y}**. A day-ahead forecast only needs consumption "
                f"up to two days before, so the model can forecast up to **{service.last_forecastable_day:%d %B %Y}**. "
                "Those days have no observed weather: the temperature defaults to the average of past years for "
                "that date. Shift it to see how demand reacts.")
    c1, c2 = st.columns([1, 2])
    days = [service.last_forecastable_day - timedelta(days=1), service.last_forecastable_day]
    day = c1.radio("Day", days, format_func=lambda d: f"{d:%A %d %B %Y}")
    offset = c2.slider("Temperature compared with a normal day (°C)", -8.0, 8.0, 0.0, 0.5)
    f = service.forecast(day, temp_offset=offset)
    base = service.forecast(day) if offset else f
    m = st.columns(4)
    m[0].metric("Energy over the day", f"{f['daily_energy_gwh']:,.0f} GWh",
                delta=f"{f['daily_energy_gwh'] - base['daily_energy_gwh']:+,.0f} GWh" if offset else None,
                delta_color="inverse")
    m[1].metric(f"Peak (at {f['peak_hour']:02d}:00)", f"{f['peak_mw'] / 1000:.1f} GW")
    m[2].metric("Mean temperature", f"{np.mean([r['temperature'] for r in f['hourly']]):.1f} °C")
    m[3].metric("Scenario", "Normal weather" if not offset else f"{offset:+.1f} °C")
    st.plotly_chart(day_chart(f["hourly"], f"Forecast for {day:%A %d %B %Y}", show_rte=False), use_container_width=True)
    st.caption("A one-day temperature change moves this forecast less than a lasting cold spell would: the model "
               "anchors on recent real consumption, which a longer spell would raise too. The heating curve in "
               "'What drives demand' shows the long-run effect (about 2,100 MW per °C in winter).")

# ---------------------------------------------------------------- backtest
with tab_bt:
    st.header("Backtest: 7.5 years of day-ahead forecasts")
    st.caption("Expanding window: each year is forecast by models trained only on the years before it. "
               "MAPE = mean absolute percentage error over every hour.")
    o = bt["overall"]
    rows = [{"Model": names[k], "MAPE": o[k]["mape"], "Average error (MW)": o[k]["mae_mw"],
             "Bias (MW)": o[k]["bias_mw"]} for k in names]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "MAPE": st.column_config.NumberColumn(format="percent"),
        "Average error (MW)": st.column_config.NumberColumn(format="%.0f"),
        "Bias (MW)": st.column_config.NumberColumn(format="%+.0f")})
    st.markdown(
        f"""<div class="callout"><b>Honest reading.</b> WattCast ({o['lightgbm']['mape']:.2%}) is far better than
        the baselines and slightly better than RTE's <i>published</i> forecast ({o['rte']['mape']:.2%}), but that
        comparison flatters it: since 2023 RTE's published forecast sits about 1.1 GW below published consumption,
        even for its same-day forecast, which suggests the two series are not defined identically. Removing that bias
        with information available at the time gives RTE <b>{o['rte_debiased']['mape']:.2%}</b>, better than
        WattCast. RTE also uses real weather forecasts, while this backtest uses observed temperatures, see below.</div>""",
        unsafe_allow_html=True)
    st.image(str(FIGURES_DIR / "mape_by_year.png"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES_DIR / "frozen_vs_retrained.png"), use_container_width=True)
    t = meta["temperature_forecast_error"]
    c2.markdown(f"""
**Lessons**

- **Retrain after a shock.** A model trained on 2015-2021 and never updated over-forecasts 2023 by
  ~{meta['frozen_model'][1]['bias_mw_frozen']:,.0f} MW on average: the energy crisis and the
  *sobriété* plan lowered consumption. Retraining every year cuts its error from
  {meta['frozen_model'][1]['mape_frozen']:.2%} to {[r for r in bt['by_year'] if r['year'] == 2023][0]['lightgbm']:.2%}.
- **Weather forecasts cost accuracy.** The backtest uses observed temperatures, i.e. a perfect weather forecast.
  Adding a realistic {t['temp_error_sigma_c']} °C day-ahead temperature error raises the {t['year']} error from
  {t['mape_observed_temp']:.2%} to **{t['mape_noisy_temp']:.2%}**.
- **Intervals must be calibrated.** Raw quantile models covered only ~64% of hours instead of 80%. Conformal
  calibration on a held-out year brings coverage to **{bt['interval_coverage_80']:.0%}**.
""")
    by_year = pd.DataFrame(bt["by_year"])
    show = by_year[["year", "lightgbm", "rte", "rte_debiased", "linear", "naive_7d", "interval_coverage"]].rename(columns={
        "year": "Year", "lightgbm": names["lightgbm"], "rte": names["rte"], "rte_debiased": names["rte_debiased"],
        "linear": names["linear"], "naive_7d": names["naive_7d"], "interval_coverage": "80% interval coverage"})
    st.caption("Interval coverage varies by year (it was too narrow in 2022, the crisis year, and too wide in 2023) but averages 80%.")
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in show.columns if c != "Year"}
                 | {"Year": st.column_config.NumberColumn(format="%d")})
    worst = pd.DataFrame(bt["worst_days"]).rename(columns={"day": "Day", "mape": "WattCast MAPE", "rte_mape": "RTE MAPE",
                                                            "temp": "Mean °C"})
    st.subheader("The 10 hardest days")
    st.caption("All ten are from the first COVID lockdown (March-May 2020). Consumption fell by over 10% almost "
               "overnight; a model that learns from consumption 2 and 7 days earlier needs days to catch up. RTE's "
               "forecast stayed within 2-6% on those days. No model trained on history alone can foresee a lockdown.")
    st.dataframe(worst, hide_index=True, use_container_width=True, column_config={
        "WattCast MAPE": st.column_config.NumberColumn(format="percent"),
        "RTE MAPE": st.column_config.NumberColumn(format="percent"),
        "Mean °C": st.column_config.NumberColumn(format="%.1f")})

# ----------------------------------------------------------------- drivers
with tab_drivers:
    st.header("What drives French electricity demand")
    c1, c2 = st.columns([3, 2])
    c1.image(str(FIGURES_DIR / "heating_curve.png"), use_container_width=True)
    th = meta["thermosensitivity"]
    c2.markdown(f"""
**Temperature is the biggest physical driver.** About a third of French homes heat with electricity, so below ~15 °C every
degree colder adds roughly **{th[0]['mw_per_degree']:,.0f} MW** (2015-2019) and **{th[1]['mw_per_degree']:,.0f} MW**
(2023-2026), on working days. RTE quotes about 2,400 MW per °C: the same order of magnitude, measured
here independently.

**The whole curve moved down after 2022.** At the same temperature, France now uses several GW less than before
the energy crisis.

**Above ~22 °C demand rises again** (air conditioning), but far less steeply than in winter.
""")
    fi = pd.Series(meta["feature_importance"]).head(12)[::-1].rename(index=FEATURE_LABELS)
    fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=C_OURS))
    fig.update_layout(title="What the model relies on (share of LightGBM gain)", xaxis_tickformat=".0%")
    st.caption("The model leans most on consumption 2 and 7 days earlier: those values already contain the weather, the season and the economy of the moment. The temperature features then correct for how this day differs from them.")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    by_hour = pd.Series(bt["mape_by_hour"]).sort_index()
    fig = go.Figure(go.Bar(x=[f"{int(h):02d}:00" for h in by_hour.index], y=by_hour.values, marker_color=C_OURS))
    fig.update_layout(title="Forecast error by hour of the day", yaxis_tickformat=".1%")
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)

# --------------------------------------------------------------------- how
with tab_how:
    st.header("How it works")
    st.markdown("""
**The task.** Forecast the 24 hourly values of French national consumption for day D, the day before (like RTE's
day-ahead forecast, used by the electricity market and to schedule power plants). The forecast is made on D-1, so
only consumption up to D-2 is used.

**Data.**
- RTE eCO2mix national consumption (30-min, averaged to hourly), 2015 to June 2026, plus RTE's own forecasts for comparison.
- Open-Meteo hourly temperature for Paris, Lyon, Marseille, Lille and Bordeaux, weighted by metropolitan population.

**Features.**
- Calendar: hour, weekday, season, French public holidays, bridge days (*ponts*), August, Christmas.
- Temperature with thermal inertia (24 h and 72 h smoothing), heating and cooling degrees.
- Consumption 2 days and 7 days earlier, and how much warmer or colder it is now than then.

**Models.**
- Two baselines: the same hour last week, and a linear regression.
- LightGBM for the forecast, plus two quantile models for the 80% interval, conformally calibrated.
- Every model is backtested year by year, never on data it was trained on.

**Limitations.**
- Temperatures in the backtest are observed, not forecast. The realistic cost of that is measured above.
- National level only, with no weather beyond temperature (wind, sun, cloud cover).
- No knowledge of exceptional events (strikes, a lockdown) until they show up in the lags.
""")
