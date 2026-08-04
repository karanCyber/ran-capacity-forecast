"""Tests. The leakage guard is the one that matters.

Everything else here is a sanity check; ``test_no_leakage_from_recent_hours``
is the test that would catch the single most damaging silent bug in a
forecasting pipeline -- a feature that is unavailable at forecast time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ran_forecast.anomaly import clean_history, detect, fit_scales, group_alerts
from ran_forecast.config import CONFIG, Config
from ran_forecast.data import generate, ingest
from ran_forecast.models.baseline import mase, score, seasonal_naive
from ran_forecast.models.features import (
    LAGS,
    MIN_LAG,
    RAW_BASELINE_COLUMN,
    build_features,
    training_frame,
)


@pytest.fixture(scope="module")
def small_cfg() -> Config:
    from dataclasses import replace

    return replace(CONFIG, n_cells=4, n_days=40, random_seed=7)


@pytest.fixture(scope="module")
def hourly(small_cfg: Config) -> pd.DataFrame:
    raw, _ = generate.generate(small_cfg)
    return ingest.to_hourly(ingest.clean_raw(raw, small_cfg), small_cfg)


# ---------------------------------------------------------------------------
# The important one
# ---------------------------------------------------------------------------

def test_all_lags_are_at_least_24h():
    """Any lag under 24h is unusable for a day-ahead forecast."""
    assert MIN_LAG >= 24
    assert min(LAGS) >= 24


def test_no_leakage_from_recent_hours(hourly: pd.DataFrame):
    """A row's features must not depend on the 24 hours immediately before it.

    Take the final 24-hour window W = [t_max-23h, t_max]. For any target row
    ``t`` inside W, every feature must be drawn from ``t - 24h`` or earlier,
    which is strictly before W begins. So corrupting the whole of W must leave
    the feature rows *inside* W bit-identical.

    This is the test that fails the moment someone adds a ``lag_1``: at forecast
    origin you do not have last hour's counter for tomorrow evening's forecast.
    """
    original = build_features(hourly)

    window_start = hourly["timestamp"].max() - pd.Timedelta(hours=23)
    corrupted_input = hourly.copy()
    in_window = corrupted_input["timestamp"] >= window_start
    corrupted_input.loc[in_window, "prb_util"] = 999.0
    corrupted = build_features(corrupted_input)

    feature_cols = [f"lag_{l}" for l in LAGS] + [
        "roll_mean_24", "roll_mean_168", "roll_std_24", "roll_max_24",
        "week_over_week", "day_over_day",
    ]
    target = original["timestamp"] >= window_start
    assert target.sum() > 0

    left = original.loc[target.to_numpy(), feature_cols].reset_index(drop=True)
    right = corrupted.loc[target.to_numpy(), feature_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_future_rows_get_complete_features(hourly: pd.DataFrame):
    """Rows beyond the end of history must still have fillable lags.

    This is the leakage constraint viewed from the serving side: if the feature
    set could not be built for an unobserved hour, the model could not be used
    in production at all.
    """
    from ran_forecast.pipeline import extend_with_future

    extended = build_features(clean_history(extend_with_future(hourly, 24)))
    future = extended[extended["prb_util"].isna()]
    assert len(future) == hourly["cell_id"].nunique() * 24
    for lag in (24, 48, 168):
        assert future[f"lag_{lag}"].notna().all(), f"lag_{lag} unfillable for future rows"


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def test_out_of_range_values_are_voided_not_clipped(small_cfg: Config):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="10min", tz="UTC"),
            "cell_id": "CELL_0000",
            "site_id": "SITE_0000",
            "archetype": "mixed",
            "prb_util": [10.0, 999.0, 12.0, -5.0, 14.0, 15.0],
            "is_injected": False,
        }
    )
    cleaned = ingest.clean_raw(raw, small_cfg)
    assert cleaned["prb_util"].max() <= 100.0
    assert cleaned["prb_util"].min() >= 0.0
    # 999 must not survive as 100: it was interpolated between neighbours.
    assert not (cleaned["prb_util"] == 999.0).any()
    assert not (cleaned["prb_util"] == 100.0).any()


def test_hourly_resample_shape(hourly: pd.DataFrame, small_cfg: Config):
    assert hourly["cell_id"].nunique() == small_cfg.n_cells
    per_cell = hourly.groupby("cell_id").size()
    assert per_cell.nunique() == 1, "cells must share one contiguous hourly grid"
    assert hourly["prb_util"].between(0, 100).all()


def test_prb_is_bounded(hourly: pd.DataFrame):
    assert hourly["prb_util"].min() >= 0.0
    assert hourly["prb_util"].max() <= 100.0


# ---------------------------------------------------------------------------
# Baseline and metrics
# ---------------------------------------------------------------------------

def test_seasonal_naive_is_exactly_168h_back(hourly: pd.DataFrame):
    naive = seasonal_naive(hourly)
    df = hourly.sort_values(["cell_id", "timestamp"]).copy()
    df["naive"] = naive.to_numpy()
    one = df[df["cell_id"] == df["cell_id"].iloc[0]].reset_index(drop=True)
    assert np.isnan(one.loc[167, "naive"])
    assert one.loc[168, "naive"] == pytest.approx(one.loc[0, "prb_util"])


def test_mase_is_one_for_the_baseline_itself():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 18.0, 33.0, 37.0])
    naive_mae = float(np.mean(np.abs(actual - predicted)))
    assert mase(actual, predicted, naive_mae) == pytest.approx(1.0)


def test_score_clips_predictions_into_valid_range():
    actual = np.array([50.0, 50.0])
    result = score(actual, np.array([-40.0, 180.0]), naive_mae=1.0)
    # Clipped to 0 and 100 -> errors of 50 each, not 90 and 130.
    assert result["mae"] == pytest.approx(50.0)


def test_mape_floor_excludes_low_utilisation_hours():
    from ran_forecast.models.baseline import mape

    actual = np.array([0.5, 60.0])
    predicted = np.array([1.5, 63.0])
    # The 0.5 -> 1.5 miss is 200% error and dominates the unfiltered figure.
    assert mape(actual, predicted) > 100
    assert mape(actual, predicted, floor=20.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _residual_frame(residuals: list[float]) -> pd.DataFrame:
    n = len(residuals)
    return pd.DataFrame(
        {
            "cell_id": "CELL_0000",
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"),
            "prb_util": 50.0,
            "yhat_model": 50.0 - np.array(residuals),
            "residual_model": residuals,
            "is_injected": False,
        }
    )


def test_single_hour_breach_is_not_an_alert():
    """One breach must not fire; the confirmation rule exists for a reason."""
    residuals = [0.4, -0.3, 0.2, 40.0, 0.1, -0.2] * 6
    flagged = detect(_residual_frame(residuals))
    assert not flagged["is_anomaly"].any()


def test_two_consecutive_breaches_do_alert():
    residuals = ([0.4, -0.3, 0.2, -0.1] * 10) + [40.0, 42.0] + ([0.2, -0.2] * 5)
    flagged = detect(_residual_frame(residuals))
    assert flagged["is_anomaly"].sum() >= 2


def test_mad_scale_resists_contamination():
    """A handful of huge outliers must not inflate the scale and hide themselves."""
    clean = [0.5, -0.4, 0.3, -0.2] * 24
    contaminated = clean.copy()
    for i in range(0, 12):
        contaminated[i] = 60.0

    scale_clean = fit_scales(_residual_frame(clean))["scale"].median()
    scale_dirty = fit_scales(_residual_frame(contaminated))["scale"].median()
    assert scale_dirty < scale_clean * 3, "MAD scale was dragged up by outliers"


def test_history_cleaning_repairs_an_injected_outage():
    """A cleaned outage must not propagate into next week's lag features."""
    n = 24 * 28
    index = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    hour = index.hour.to_numpy()
    values = 40.0 + 20.0 * np.sin(2 * np.pi * hour / 24)

    df = pd.DataFrame(
        {
            "cell_id": "CELL_0000",
            "site_id": "SITE_0000",
            "archetype": "mixed",
            "timestamp": index,
            "prb_util": values,
            "n_samples": 6,
            "imputed": False,
            "is_injected": False,
        }
    )
    outage = slice(400, 408)
    df.loc[outage, "prb_util"] = 1.0

    cleaned = clean_history(df)
    assert cleaned.loc[outage, "hist_anomaly"].all()
    # Repaired values should be back near the normal profile, not near 1.0.
    assert (cleaned.loc[outage, "prb_clean"] > 10.0).all()


def test_alert_episodes_collapse_consecutive_hours():
    residuals = ([0.3, -0.2] * 20) + [40.0, 41.0, 42.0, 43.0] + ([0.3, -0.2] * 20)
    flagged = detect(_residual_frame(residuals))
    episodes = group_alerts(flagged)
    assert len(episodes) == 1
    assert episodes["hours"].iloc[0] >= 3


# ---------------------------------------------------------------------------
# Training frame integrity
# ---------------------------------------------------------------------------

def test_training_frame_has_no_missing_features(hourly: pd.DataFrame):
    frame = training_frame(build_features(clean_history(hourly)))
    assert len(frame) > 0
    for lag in LAGS:
        assert frame[f"lag_{lag}"].notna().all()
    assert frame[RAW_BASELINE_COLUMN].notna().all()
    assert not frame["imputed"].any()
