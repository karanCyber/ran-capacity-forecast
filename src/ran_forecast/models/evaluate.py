"""Rolling-origin backtest.

A single train/test split on a time series with a growth trend flatters
whichever model happens to suit the last three weeks. Instead we walk the
forecast origin forward through the evaluation window:

    origin T  ->  train on everything <= T, forecast T+1 .. T+24
    origin T+24 -> ...

Both the baseline and LightGBM are scored on *exactly the same rows*, which is
the only way the side-by-side table means anything.

Refitting: the model is refit every ``refit_every_days`` (default 7) rather than
at every origin. Refitting daily is more rigorous but ~7x the compute for a
difference well inside the noise here, and weekly retraining is closer to how
this would actually run in production. The nightly CronJob in ``k8s/`` retrains
on the full history; this knob only controls backtest cost.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ran_forecast.config import CONFIG, Config
from ran_forecast.models import lgbm
from ran_forecast.models.baseline import score
from ran_forecast.anomaly import clean_history
from ran_forecast.models.features import (
    BASELINE_COLUMN,
    RAW_BASELINE_COLUMN,
    build_features,
    seasonal_naive_mae,
    training_frame,
)

log = logging.getLogger(__name__)


def _origins(timestamps: pd.DatetimeIndex, cfg: Config) -> list[pd.Timestamp]:
    """Daily forecast origins covering the last ``backtest_days``."""
    end = timestamps.max()
    last_origin = end - pd.Timedelta(hours=cfg.horizon_hours)
    first_origin = last_origin - pd.Timedelta(days=cfg.backtest_days - 1)
    return list(pd.date_range(first_origin, last_origin, freq="24h"))


def backtest(
    hourly: pd.DataFrame,
    cfg: Config = CONFIG,
    refit_every_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """Return per-row predictions for both models, plus the metrics dict."""
    features = build_features(clean_history(hourly))
    timestamps = pd.DatetimeIndex(features["timestamp"].unique()).sort_values()
    origins = _origins(timestamps, cfg)
    log.info("backtest: %d origins from %s to %s", len(origins), origins[0], origins[-1])

    booster = meta = None
    naive_mae_train = np.nan
    last_fit: pd.Timestamp | None = None
    chunks: list[pd.DataFrame] = []

    for i, origin in enumerate(origins):
        needs_refit = last_fit is None or (origin - last_fit) >= pd.Timedelta(days=refit_every_days)
        if needs_refit:
            train_rows = training_frame(features[features["timestamp"] <= origin])
            booster, meta = lgbm.train(train_rows)
            naive_mae_train = seasonal_naive_mae(train_rows)
            last_fit = origin
            log.info("  [%2d/%d] refit at %s on %d rows (naive MAE %.3f)",
                     i + 1, len(origins), origin, len(train_rows), naive_mae_train)

        window = features[
            (features["timestamp"] > origin)
            & (features["timestamp"] <= origin + pd.Timedelta(hours=cfg.horizon_hours))
        ]
        # Score only on rows with a usable baseline and a genuine observation.
        window = window[
            window[BASELINE_COLUMN].notna()
            & window[RAW_BASELINE_COLUMN].notna()
            & ~window["imputed"]
        ]
        window = window.dropna(subset=[f"lag_{l}" for l in (24, 48, 168, 336)])
        if window.empty:
            continue

        chunk = window[
            ["cell_id", "site_id", "archetype", "timestamp", "prb_util", "is_injected"]
        ].copy()
        chunk["origin"] = origin
        chunk["horizon"] = (
            (chunk["timestamp"] - origin).dt.total_seconds() // 3600
        ).astype(int)
        chunk["yhat_baseline"] = window[RAW_BASELINE_COLUMN].to_numpy()
        chunk["yhat_baseline_clean"] = window[BASELINE_COLUMN].to_numpy()
        chunk["yhat_model"] = lgbm.predict(booster, window, meta)
        chunks.append(chunk)

    predictions = pd.concat(chunks, ignore_index=True)
    actual = predictions["prb_util"].to_numpy(dtype=float)

    metrics = {
        "naive_mae_in_sample": round(float(naive_mae_train), 4),
        "origins": len(origins),
        "refit_every_days": refit_every_days,
        "horizon_hours": cfg.horizon_hours,
        "eval_start": str(predictions["timestamp"].min()),
        "eval_end": str(predictions["timestamp"].max()),
        "seasonal_naive": score(actual, predictions["yhat_baseline"].to_numpy(), naive_mae_train),
        "seasonal_naive_cleaned": score(
            actual, predictions["yhat_baseline_clean"].to_numpy(), naive_mae_train
        ),
        "lightgbm": score(actual, predictions["yhat_model"].to_numpy(), naive_mae_train),
    }

    improvement = 100 * (
        1 - metrics["lightgbm"]["mae"] / metrics["seasonal_naive"]["mae"]
    )
    metrics["mae_improvement_pct"] = round(float(improvement), 2)

    predictions["residual_baseline"] = actual - predictions["yhat_baseline"]
    predictions["residual_baseline_clean"] = actual - predictions["yhat_baseline_clean"]
    predictions["residual_model"] = actual - predictions["yhat_model"]
    return predictions, metrics


def metrics_by_segment(predictions: pd.DataFrame, naive_mae: float, by: str) -> pd.DataFrame:
    """Break the headline numbers down by horizon, archetype, hour, etc."""
    rows = []
    for key, group in predictions.groupby(by, observed=True):
        actual = group["prb_util"].to_numpy(dtype=float)
        base = score(actual, group["yhat_baseline"].to_numpy(), naive_mae)
        model = score(actual, group["yhat_model"].to_numpy(), naive_mae)
        rows.append(
            {
                by: key,
                "n": base["n"],
                "baseline_mae": base["mae"],
                "model_mae": model["mae"],
                "baseline_rmse": base["rmse"],
                "model_rmse": model["rmse"],
                "model_mase": model["mase"],
                "improvement_pct": round(100 * (1 - model["mae"] / base["mae"]), 2)
                if base["mae"]
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(by).reset_index(drop=True)
