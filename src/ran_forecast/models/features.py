"""Feature engineering for the day-ahead forecast.

The single most important constraint here
-----------------------------------------
We forecast up to 24 hours ahead. At forecast origin ``T`` we know the series up
to and including ``T``. Predicting ``y(T+1)`` with ``lag_1 = y(T)`` looks great
in a backtest and is unusable in production, because at 08:00 you do not have
the 22:00 reading you would need for the 23:00 forecast.

So **every lag is at least 24 hours**. Concretely, for any target timestamp
``t = T + h`` with ``h`` in 1..24, ``lag_24 = y(t - 24)`` is at or before ``T``.
Rolling statistics are shifted by 24 hours for the same reason.

A consequence worth stating explicitly: because all features are >= 24h old,
the feature set is *horizon-invariant* -- the row used to predict hour T+1 is
built the same way as the row for hour T+24. Adding an explicit ``horizon``
column would therefore carry no information. The tradeoff is that the model
does not exploit the extra freshness available at short horizons; it is a
uniform 24-hour-ahead model. That is the honest reading of these features.

Why there is no time-index / trend feature
------------------------------------------
Gradient-boosted trees cannot extrapolate. Handing them ``days_since_start``
would let them fit the rollout trend in-sample and then flatline at the last
seen value out-of-sample. The trend is carried implicitly by the recent lags
instead, and by the target definition below.

Target definition
-----------------
The model predicts the *delta from the seasonal-naive baseline*::

    target = y(t) - y(t - 168)

so the final forecast is ``seasonal_naive + model_correction``. This frames
LightGBM as "learn what the baseline gets wrong", keeps the level and the
growth trend outside the tree ensemble where they belong, and means a model
that has learned nothing scores MASE ~= 1.0 rather than something arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ran_forecast.config import DAILY_LAG_HOURS, WEEKLY_LAG_HOURS

LAGS = (24, 25, 26, 27, 48, 72, 168, 169, 170, 336)
ROLL_WINDOWS = (24, 168)
MIN_LAG = min(LAGS)

CATEGORICAL = ["archetype", "cell_id"]

FEATURE_COLUMNS: list[str] = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["roll_max_24", "week_over_week", "day_over_day", "hour", "dayofweek", "is_weekend"]
    + CATEGORICAL
)

TARGET = "target_delta"
# The model's anchor: seasonal naive computed on the anomaly-cleaned history.
BASELINE_COLUMN = f"lag_{WEEKLY_LAG_HOURS}"
# The honest baseline for the results table: seasonal naive on raw history,
# echoed outages and all. This is what "no model at all" actually gets you.
RAW_BASELINE_COLUMN = "sn_raw"


def build_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Attach lag/rolling/calendar features. Input must be sorted per cell.

    Lags are taken from ``prb_clean`` when present (see
    ``anomaly.clean_history``) so that a past outage does not propagate into
    next week's inputs. The target is always the raw observation.
    """
    df = hourly.sort_values(["cell_id", "timestamp"]).copy()
    source = "prb_clean" if "prb_clean" in df.columns else "prb_util"
    df[RAW_BASELINE_COLUMN] = df.groupby("cell_id", sort=False)["prb_util"].shift(
        WEEKLY_LAG_HOURS
    )
    grouped = df.groupby("cell_id", sort=False)[source]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Shift by 24 first, then roll: the window ends 24h before the target, so
    # nothing inside it is newer than the forecast origin.
    shifted = grouped.shift(DAILY_LAG_HOURS)
    by_cell = shifted.groupby(df["cell_id"], sort=False)
    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}"] = by_cell.transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 4)).mean()
        )
        df[f"roll_std_{window}"] = by_cell.transform(
            lambda s, w=window: s.rolling(w, min_periods=max(2, w // 4)).std()
        )
    df["roll_max_24"] = by_cell.transform(
        lambda s: s.rolling(24, min_periods=6).max()
    )

    # Momentum terms: is this week busier than last, this day than the one before?
    df["week_over_week"] = df[f"lag_{WEEKLY_LAG_HOURS}"] - df["lag_336"]
    df["day_over_day"] = df["lag_24"] - df["lag_48"]

    ts = df["timestamp"].dt
    df["hour"] = ts.hour.astype("int16")
    df["dayofweek"] = ts.dayofweek.astype("int16")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    df[TARGET] = df["prb_util"] - df[BASELINE_COLUMN]
    return df


def training_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting: complete features, real (non-imputed) target."""
    required = [f"lag_{l}" for l in LAGS] + [TARGET, RAW_BASELINE_COLUMN]
    mask = features[required].notna().all(axis=1) & ~features["imputed"]
    return features.loc[mask]


def seasonal_naive_mae(frame: pd.DataFrame) -> float:
    """Denominator for MASE: in-sample MAE of the **raw** seasonal naive.

    Anchoring on the raw baseline (rather than the cleaned one used as the
    model's input) is what makes MASE readable: 1.0 means "no better than
    same-hour-last-week with no pipeline at all", which is the comparison the
    results table is claiming to win.
    """
    errors = (frame["prb_util"] - frame[RAW_BASELINE_COLUMN]).abs()
    return float(errors.mean())


def clip_prb(values: np.ndarray | pd.Series) -> np.ndarray:
    """PRB utilisation is a bounded percentage; forecasts must respect that."""
    return np.clip(np.asarray(values, dtype=float), 0.0, 100.0)
