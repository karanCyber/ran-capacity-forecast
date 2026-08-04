"""End-to-end orchestration: train, forecast forward, build the serving artifact.

The API never trains and never runs inference. It loads a precomputed parquet
file. That keeps request latency flat and predictable, makes the container
trivially horizontally scalable, and means a bad model rollout is a file swap
rather than a redeploy.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ran_forecast.anomaly import (
    CAPACITY_RISK_THRESHOLD,
    clean_history,
    detect,
    eligible_events,
    fit_scales,
    group_alerts,
    score_episodes,
    score_eventwise,
    score_pointwise,
    sweep_threshold,
)
from ran_forecast.config import CONFIG, Config
from ran_forecast.models import lgbm
from ran_forecast.models.evaluate import backtest, metrics_by_segment
from ran_forecast.models.features import (
    BASELINE_COLUMN,
    RAW_BASELINE_COLUMN,
    build_features,
    seasonal_naive_mae,
    training_frame,
)

log = logging.getLogger(__name__)


def extend_with_future(hourly: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Append ``horizon_hours`` empty rows per cell after the last observation.

    Feature construction then fills their lags from real history. This works
    only because every lag is >= 24h; with a ``lag_1`` feature these rows would
    be unfillable, which is the same constraint seen from the other side.
    """
    last = hourly["timestamp"].max()
    future_index = pd.date_range(
        last + pd.Timedelta(hours=1), periods=horizon_hours, freq="1h", name="timestamp"
    )
    static = hourly.groupby("cell_id")[["site_id", "archetype"]].first().reset_index()

    future = static.merge(pd.DataFrame({"timestamp": future_index}), how="cross")
    future["prb_util"] = np.nan
    future["n_samples"] = 0
    future["imputed"] = False
    future["is_injected"] = False

    combined = pd.concat([hourly, future[hourly.columns]], ignore_index=True)
    return combined.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)


def train_full(hourly: pd.DataFrame, cfg: Config = CONFIG):
    """Fit on all available history and persist the model."""
    features = build_features(clean_history(hourly))
    rows = training_frame(features)
    booster, meta = lgbm.train(rows)
    meta["naive_mae_in_sample"] = round(seasonal_naive_mae(rows), 4)
    meta["trained_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    lgbm.save(booster, meta, cfg.model_path, cfg.model_meta_path)
    log.info("model saved to %s", cfg.model_path)
    return booster, meta


def forecast_future(hourly: pd.DataFrame, booster, meta, cfg: Config = CONFIG) -> pd.DataFrame:
    """Genuine out-of-sample forecast for the next ``horizon_hours``."""
    extended = build_features(clean_history(extend_with_future(hourly, cfg.horizon_hours)))
    future = extended[extended["prb_util"].isna()].copy()
    future = future[future[BASELINE_COLUMN].notna()]
    if future.empty:
        log.warning("no future rows had a usable baseline; skipping forward forecast")
        return pd.DataFrame()

    future["yhat_model"] = lgbm.predict(booster, future, meta)
    future["yhat_baseline"] = future[RAW_BASELINE_COLUMN]
    future["capacity_risk"] = future["yhat_model"] >= CAPACITY_RISK_THRESHOLD
    future["is_forecast"] = True
    future["is_anomaly"] = False
    future["severity"] = "none"
    future["z_score"] = np.nan
    log.info("forward forecast: %d rows, %d flagged as capacity risk",
             len(future), int(future["capacity_risk"].sum()))
    return future


SERVING_COLUMNS = [
    "cell_id", "site_id", "archetype", "timestamp", "prb_util",
    "yhat_baseline", "yhat_model", "z_score", "is_anomaly", "severity",
    "direction", "capacity_risk", "is_forecast",
]


def build_serving_artifact(
    hourly: pd.DataFrame,
    cfg: Config = CONFIG,
    refit_every_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the full evaluation and produce everything the API serves."""
    predictions, metrics = backtest(hourly, cfg, refit_every_days=refit_every_days)

    flagged = detect(predictions, cfg)
    truth_path = cfg.injected_anomalies_path
    truth = (
        pd.read_csv(truth_path, parse_dates=["start", "end"])
        if truth_path.exists()
        else pd.DataFrame()
    )

    if len(truth):
        events = eligible_events(truth, flagged)
        metrics["anomaly"] = {
            "k": cfg.anomaly_k,
            "consecutive": cfg.anomaly_consecutive,
            "pointwise": score_pointwise(flagged),
            "episodes": score_episodes(flagged, events),
            "event_recall": score_eventwise(flagged, events).to_dict(orient="records"),
            "threshold_sweep": sweep_threshold(predictions, events, cfg=cfg).to_dict(
                orient="records"
            ),
        }

    booster, meta = train_full(hourly, cfg)
    metrics["model_meta"] = {
        key: meta[key] for key in ("n_train_rows", "train_end", "trained_at")
    }
    metrics["feature_importance"] = lgbm.feature_importance(booster).to_dict(orient="records")

    naive_mae = metrics["naive_mae_in_sample"]
    metrics["by_horizon"] = metrics_by_segment(predictions, naive_mae, "horizon").to_dict(
        orient="records"
    )
    metrics["by_archetype"] = metrics_by_segment(predictions, naive_mae, "archetype").to_dict(
        orient="records"
    )

    future = forecast_future(hourly, booster, meta, cfg)

    flagged["is_forecast"] = False
    parts = [flagged.reindex(columns=SERVING_COLUMNS)]
    if not future.empty:
        future["direction"] = "n/a"
        parts.append(future.reindex(columns=SERVING_COLUMNS))
    serving = pd.concat(parts, ignore_index=True).sort_values(["cell_id", "timestamp"])

    episodes = group_alerts(flagged)
    return serving.reset_index(drop=True), episodes, metrics


def write_artifacts(
    serving: pd.DataFrame, episodes: pd.DataFrame, metrics: dict, cfg: Config = CONFIG
) -> None:
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    serving.to_parquet(cfg.forecasts_path, index=False)
    episodes.to_parquet(cfg.artifact_dir / "episodes.parquet", index=False)
    cfg.metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("wrote %s (%d rows)", cfg.forecasts_path, len(serving))
    log.info("wrote %s (%d episodes)", cfg.artifact_dir / "episodes.parquet", len(episodes))
    log.info("wrote %s", cfg.metrics_path)
