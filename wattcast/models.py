"""Forecasting models: two baselines, a linear model and LightGBM (point + 80% interval)."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import FEATURES

LGB_PARAMS = dict(n_estimators=900, learning_rate=0.04, num_leaves=63, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1, n_jobs=2)
LINEAR_NUMERIC = ["hdd", "cdd", "temp_smooth_72h", "load_lag_7d", "load_day_mean_lag_2d", "temp_minus_lag_7d",
                  "is_holiday", "is_bridge", "august", "xmas_break", "years_since_2015"]


def make_linear():
    pre = ColumnTransformer([
        ("calendar", OneHotEncoder(handle_unknown="ignore"), ["hour", "dow"]),
        ("numeric", StandardScaler(), LINEAR_NUMERIC),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


@dataclass
class Forecaster:
    """LightGBM point forecast plus an 80% prediction interval.

    The interval comes from 10% / 90% quantile models, then *conformally calibrated*: quantile models fitted on
    the training data are over-confident on new data, so the last year of training is held out, the interval is
    widened until it covers 80% of that year, and that widening is applied to future forecasts.
    """
    point: lgb.LGBMRegressor = field(default_factory=lambda: lgb.LGBMRegressor(**LGB_PARAMS))
    low: lgb.LGBMRegressor = field(default_factory=lambda: lgb.LGBMRegressor(objective="quantile", alpha=0.1, **{**LGB_PARAMS, "n_estimators": 400}))
    high: lgb.LGBMRegressor = field(default_factory=lambda: lgb.LGBMRegressor(objective="quantile", alpha=0.9, **{**LGB_PARAMS, "n_estimators": 400}))
    coverage: float = 0.8
    calibration_hours: int = 365 * 24
    interval_margin: float = 0.0
    trained_until: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster":
        Xf = X[FEATURES]
        fit_part, cal = slice(None, -self.calibration_hours), slice(-self.calibration_hours, None)
        self.low.fit(Xf.iloc[fit_part], y.iloc[fit_part])
        self.high.fit(Xf.iloc[fit_part], y.iloc[fit_part])
        lo, hi = self.low.predict(Xf.iloc[cal]), self.high.predict(Xf.iloc[cal])
        yc = y.iloc[cal].to_numpy()
        scores = np.maximum(lo - yc, yc - hi)  # > 0 when the true value falls outside the interval
        n = len(scores)
        self.interval_margin = float(np.quantile(scores, min(1.0, self.coverage * (1 + 1 / n))))
        # Refit on everything so the interval knows the latest consumption level; the margin learned above is kept.
        self.low.fit(Xf, y)
        self.high.fit(Xf, y)
        self.point.fit(Xf, y)
        self.trained_until = str(X.index.max())
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[FEATURES]
        point = self.point.predict(Xf)
        low = self.low.predict(Xf) - self.interval_margin
        high = self.high.predict(Xf) + self.interval_margin
        # Quantile models are fitted separately and can cross: keep the point forecast inside the interval.
        return pd.DataFrame({"forecast": point, "low": np.minimum(low, point), "high": np.maximum(high, point)}, index=X.index)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    err = pred - y
    return {
        "mape": float(np.mean(np.abs(err) / y)),
        "mae_mw": float(np.mean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.mean(err ** 2))),
        "bias_mw": float(np.mean(err)),
        "n_hours": int(ok.sum()),
    }
