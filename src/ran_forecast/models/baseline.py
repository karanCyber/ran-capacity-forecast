"""Seasonal-naive baseline and the shared metric set.

The baseline is deliberately the first thing built. ``y_hat(t) = y(t - 168)``
-- same hour, one week ago -- is a strong forecaster for cellular traffic
because the weekly cycle dominates. Any model that cannot beat it is not
earning its operational cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ran_forecast.config import WEEKLY_LAG_HOURS
from ran_forecast.models.features import clip_prb


def seasonal_naive(hourly: pd.DataFrame, lag: int = WEEKLY_LAG_HOURS) -> pd.Series:
    """Same hour, ``lag`` hours ago, per cell."""
    return (
        hourly.sort_values(["cell_id", "timestamp"])
        .groupby("cell_id", sort=False)["prb_util"]
        .shift(lag)
    )


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean error. Positive means the model under-forecasts, which for capacity
    planning is the expensive direction."""
    return float(np.mean(predicted - actual))


def mape(actual: np.ndarray, predicted: np.ndarray, floor: float = 0.0) -> float:
    """Mean absolute percentage error.

    Reported for completeness only. PRB utilisation approaches zero overnight on
    business cells, so the denominator collapses and MAPE explodes on exactly
    the hours nobody cares about. ``floor`` restricts the calculation to hours
    above a utilisation threshold, which is the only form in which this number
    means anything.
    """
    mask = actual > floor
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def mase(actual: np.ndarray, predicted: np.ndarray, naive_mae: float) -> float:
    """Mean absolute scaled error, scaled by in-sample seasonal-naive MAE.

    MASE < 1 means the model beats the seasonal naive. MASE = 0.80 reads as
    "20% better than same-hour-last-week", which is the sentence the results
    table exists to produce.
    """
    if naive_mae <= 0:
        return float("nan")
    return mae(actual, predicted) / naive_mae


def score(
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_mae: float,
    mape_floor: float = 20.0,
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = clip_prb(predicted)
    return {
        "rmse": round(rmse(actual, predicted), 4),
        "mae": round(mae(actual, predicted), 4),
        "mase": round(mase(actual, predicted, naive_mae), 4),
        "bias": round(bias(actual, predicted), 4),
        "mape_all": round(mape(actual, predicted), 2),
        f"mape_above_{int(mape_floor)}": round(mape(actual, predicted, mape_floor), 2),
        "n": int(len(actual)),
    }
