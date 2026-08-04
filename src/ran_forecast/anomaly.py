"""Residual-based anomaly detection.

Method
------
1. Take the forecast residual ``e(t) = y(t) - yhat(t)`` from the LightGBM model.
2. Estimate the residual scale per cell with the **median absolute deviation**,
   not the standard deviation. Standard deviation is inflated by the very
   outliers we are hunting, so a few big spikes raise the threshold and hide
   themselves. MAD has a 50% breakdown point; with roughly 1-2% anomalous hours
   in this data the scale estimate is effectively uncontaminated.
3. Residual spread is not constant across the day -- a 3 percentage-point miss
   at busy hour is routine, at 04:00 it is not. So the scale is estimated per
   ``(cell, hour)`` and shrunk toward the cell-level scale, because 21 samples
   per hour is too thin to trust on its own.
4. Flag ``|z| > k``, then require ``consecutive`` breaches in a row before
   raising. Single-hour breaches are mostly noise; a real outage or congestion
   event persists.

Deliberately not used: isolation forests, autoencoders, changepoint libraries.
On a residual series with a known scale, a robust threshold is easier to tune,
easier to explain to an operations team, and easier to defend when it fires at
3am. Complexity here would cost trust and buy very little.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ran_forecast.config import CONFIG, Config

log = logging.getLogger(__name__)

MAD_TO_SIGMA = 1.4826
SHRINKAGE_PRIOR = 10.0  # pseudo-observations pulling hourly scale to cell scale
MIN_SCALE = 0.5  # floor, in PRB percentage points, to avoid divide-by-nothing

# A forecast at or above this utilisation is a capacity concern regardless of
# whether it is anomalous. This is the "recommendation" half of the rApp.
CAPACITY_RISK_THRESHOLD = 85.0


def clean_history(
    hourly: pd.DataFrame,
    k: float = 4.0,
    value_col: str = "prb_util",
    lag: int = 168,
) -> pd.DataFrame:
    """Repair historical anomalies so they cannot poison future lag features.

    Why this exists
    ---------------
    A seasonal-naive forecast reads ``y(t-168)``. If the cell was in outage one
    week ago, today's baseline is the outage. The model then under-forecasts by
    60 percentage points and the detector raises a large, confident, completely
    spurious alert -- exactly 168 hours after every real event. In the first
    version of this pipeline that echo was the single largest source of false
    positives.

    The fix is to build lag features from an anomaly-cleaned copy of the series.
    Detection is a first pass over history using seasonal-naive residuals; any
    hour whose residual exceeds ``k`` robust scales is replaced by the rolling
    median of the same weekday-and-hour slot across neighbouring weeks, which
    preserves both the weekly shape and the growth trend.

    The *target* is always the raw observation. Only the inputs are cleaned --
    otherwise the model would be trained to predict a fiction.
    """
    df = hourly.sort_values(["cell_id", "timestamp"]).copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    naive = df.groupby("cell_id", sort=False)[value_col].shift(lag)
    residual = df[value_col] - naive

    scale = (
        residual.groupby([df["cell_id"], df["hour"]])
        .transform(lambda s: _mad(s.to_numpy()) * MAD_TO_SIGMA)
        .clip(lower=MIN_SCALE)
    )
    flag = (residual.abs() / scale > k).fillna(False)
    # Imputed hours are already unreliable; treat them as needing repair too.
    if "imputed" in df.columns:
        flag = flag | df["imputed"].to_numpy(dtype=bool)

    masked = df[value_col].where(~flag)
    slot = masked.groupby([df["cell_id"], df["dayofweek"], df["hour"]])
    replacement = slot.transform(
        lambda s: s.rolling(5, center=True, min_periods=1).median()
    )
    replacement = replacement.fillna(slot.transform("median"))
    replacement = replacement.fillna(df.groupby("cell_id")[value_col].transform("median"))

    df["hist_anomaly"] = flag
    df["prb_clean"] = np.where(flag, replacement, df[value_col])
    log.info("history cleaning: repaired %d/%d hours (%.2f%%)",
             int(flag.sum()), len(df), 100 * float(flag.mean()))
    return df.drop(columns=["hour", "dayofweek"])


def _mad(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def fit_scales(predictions: pd.DataFrame, residual_col: str = "residual_model") -> pd.DataFrame:
    """Per (cell, hour) robust residual scale, shrunk toward the cell scale."""
    df = predictions.copy()
    df["hour"] = df["timestamp"].dt.hour

    cell_scale = (
        df.groupby("cell_id")[residual_col]
        .apply(lambda s: _mad(s.to_numpy()) * MAD_TO_SIGMA)
        .rename("cell_scale")
    )
    hour_stats = (
        df.groupby(["cell_id", "hour"])[residual_col]
        .agg(hour_mad=lambda s: _mad(s.to_numpy()), n="count")
        .reset_index()
    )
    hour_stats["hour_scale"] = hour_stats["hour_mad"] * MAD_TO_SIGMA
    hour_stats = hour_stats.merge(cell_scale, on="cell_id", how="left")

    weight = hour_stats["n"] / (hour_stats["n"] + SHRINKAGE_PRIOR)
    hour_stats["scale"] = (
        weight * hour_stats["hour_scale"].fillna(hour_stats["cell_scale"])
        + (1 - weight) * hour_stats["cell_scale"]
    ).clip(lower=MIN_SCALE)

    return hour_stats[["cell_id", "hour", "n", "scale"]]


def detect(
    predictions: pd.DataFrame,
    cfg: Config = CONFIG,
    residual_col: str = "residual_model",
    scales: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``z_score``, ``breach``, ``is_anomaly``, ``severity`` columns."""
    df = predictions.copy()
    df["hour"] = df["timestamp"].dt.hour
    if scales is None:
        scales = fit_scales(df, residual_col)

    df = df.merge(scales[["cell_id", "hour", "scale"]], on=["cell_id", "hour"], how="left")
    df["scale"] = df["scale"].fillna(df["scale"].median()).clip(lower=MIN_SCALE)
    df["z_score"] = df[residual_col] / df["scale"]
    df["breach"] = df["z_score"].abs() > cfg.anomaly_k

    # Confirmation rule: N consecutive breaches within a cell's own timeline.
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    need = max(1, cfg.anomaly_consecutive)
    rolled = (
        df.groupby("cell_id", sort=False)["breach"]
        .transform(lambda s: s.rolling(need, min_periods=need).sum())
    )
    confirmed = rolled >= need
    # Mark the whole confirming run, not just its final hour.
    is_anomaly = confirmed.copy()
    for shift in range(1, need):
        is_anomaly |= confirmed.groupby(df["cell_id"], sort=False).shift(-shift).fillna(False)
    df["is_anomaly"] = is_anomaly & df["breach"]

    df["direction"] = np.where(df["z_score"] >= 0, "above_forecast", "below_forecast")
    df["severity"] = pd.cut(
        df["z_score"].abs(),
        bins=[0, cfg.anomaly_k, cfg.anomaly_k * 1.7, cfg.anomaly_k * 3.0, np.inf],
        labels=["none", "low", "medium", "high"],
        right=False,
    ).astype(str)
    df.loc[~df["is_anomaly"], "severity"] = "none"

    df["capacity_risk"] = df["yhat_model"] >= CAPACITY_RISK_THRESHOLD

    log.info("flagged %d/%d hours as anomalous (%.2f%%)",
             int(df["is_anomaly"].sum()), len(df), 100 * df["is_anomaly"].mean())
    return df


# --------------------------------------------------------------------------
# Scoring against the injected ground truth
# --------------------------------------------------------------------------

def score_pointwise(flagged: pd.DataFrame, truth_col: str = "is_injected") -> dict:
    truth = flagged[truth_col].to_numpy(dtype=bool)
    pred = flagged["is_anomaly"].to_numpy(dtype=bool)

    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / (fp + tn), 5) if fp + tn else 0.0,
        "flagged_pct": round(100 * float(pred.mean()), 3),
    }


def score_eventwise(flagged: pd.DataFrame, truth_events: pd.DataFrame) -> pd.DataFrame:
    """An event counts as detected if any hour inside it was flagged.

    Point-level recall punishes a detector for missing the quiet tail of a slow
    drift. For an operations team what matters is whether the event surfaced at
    all, so both views are reported.
    """
    hits = flagged.loc[flagged["is_anomaly"], ["cell_id", "timestamp"]]
    rows = []
    for event in truth_events.itertuples(index=False):
        window = hits[
            (hits["cell_id"] == event.cell_id)
            & (hits["timestamp"] >= pd.Timestamp(event.start))
            & (hits["timestamp"] <= pd.Timestamp(event.end))
        ]
        rows.append({"kind": event.kind, "detected": len(window) > 0})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.groupby("kind")["detected"]
        .agg(events="count", detected="sum")
        .reset_index()
    )
    summary["recall"] = (summary["detected"] / summary["events"]).round(4)
    total = pd.DataFrame([{
        "kind": "ALL",
        "events": int(summary["events"].sum()),
        "detected": int(summary["detected"].sum()),
        "recall": round(float(summary["detected"].sum() / summary["events"].sum()), 4),
    }])
    return pd.concat([summary, total], ignore_index=True)


def group_alerts(flagged: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive flagged hours per cell into alert *episodes*.

    This is what an operations team actually receives. Point-level precision
    counts every hour separately and is misleading in both directions: a single
    8-hour outage inflates it, and one noisy hour deflates it. Episodes are the
    unit of alerting, so they are the unit of scoring.
    """
    hits = flagged.loc[flagged["is_anomaly"]].sort_values(["cell_id", "timestamp"])
    if hits.empty:
        return pd.DataFrame(
            columns=["cell_id", "start", "end", "hours", "peak_z", "direction", "severity"]
        )

    gap = hits.groupby("cell_id", sort=False)["timestamp"].diff()
    new_episode = (gap.isna()) | (gap > pd.Timedelta(hours=1))
    hits = hits.assign(episode=new_episode.cumsum())

    episodes = (
        hits.groupby(["cell_id", "episode"], sort=False)
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            hours=("timestamp", "count"),
            peak_z=("z_score", lambda s: s.iloc[np.argmax(np.abs(s.to_numpy()))]),
            mean_actual=("prb_util", "mean"),
            mean_forecast=("yhat_model", "mean"),
            overlaps_truth=("is_injected", "any"),
        )
        .reset_index(drop=False)
    )
    episodes["direction"] = np.where(episodes["peak_z"] >= 0, "above_forecast", "below_forecast")
    episodes["severity"] = pd.cut(
        episodes["peak_z"].abs(),
        bins=[0, 5.0, 10.0, np.inf],
        labels=["low", "medium", "high"],
        right=False,
    ).astype(str)
    return episodes.drop(columns=["episode"]).sort_values(
        ["peak_z"], key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def score_episodes(
    flagged: pd.DataFrame,
    truth_events: pd.DataFrame | None = None,
    tolerance_hours: int = 2,
) -> dict:
    """Episode-level precision: what fraction of raised alerts were real.

    A tolerance window is applied when matching alerts to ground truth. An
    alert that fires two hours before an outage is formally recorded is an
    early detection, not a false positive, and scoring it as one would reward
    a detector for lagging. Two hours is the smallest window that covers the
    confirmation delay built into the detector itself.
    """
    episodes = group_alerts(flagged)
    if episodes.empty:
        return {"episodes": 0, "true_episodes": 0, "episode_precision": 0.0}

    if truth_events is not None and len(truth_events):
        tol = pd.Timedelta(hours=tolerance_hours)
        events = truth_events.copy()
        events["start"] = pd.to_datetime(events["start"], utc=True) - tol
        events["end"] = pd.to_datetime(events["end"], utc=True) + tol
        by_cell: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for event in events.itertuples(index=False):
            by_cell.setdefault(event.cell_id, []).append((event.start, event.end))

        def matches(row) -> bool:
            for start, end in by_cell.get(row["cell_id"], []):
                if row["start"] <= end and row["end"] >= start:
                    return True
            return False

        episodes["overlaps_truth"] = episodes.apply(matches, axis=1)

    true_episodes = int(episodes["overlaps_truth"].sum())
    return {
        "episodes": int(len(episodes)),
        "true_episodes": true_episodes,
        "episode_precision": round(true_episodes / len(episodes), 4),
        "median_episode_hours": float(episodes["hours"].median()),
    }


def eligible_events(truth_events: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """Restrict ground truth to events overlapping the evaluation window.

    Without this, event recall is scored against events the detector never had
    the chance to see, which silently understates it by a factor of four.
    """
    lo, hi = flagged["timestamp"].min(), flagged["timestamp"].max()
    events = truth_events.copy()
    events["start"] = pd.to_datetime(events["start"], utc=True)
    events["end"] = pd.to_datetime(events["end"], utc=True)
    return events[(events["end"] >= lo) & (events["start"] <= hi)].reset_index(drop=True)


def sweep_threshold(
    predictions: pd.DataFrame,
    truth_events: pd.DataFrame | None = None,
    ks: tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
    cfg: Config = CONFIG,
) -> pd.DataFrame:
    """Precision/recall as a function of k, so the threshold choice is visible.

    Event recall is the column that decides the operating point. Point recall
    falls steeply as k rises, but that is mostly the interior hours of events
    already detected at their onset -- which produce no extra alert. If event
    recall holds flat, a higher k is buying precision essentially for free.
    """
    from dataclasses import replace

    scales = fit_scales(predictions)
    rows = []
    for k in ks:
        flagged = detect(predictions, replace(cfg, anomaly_k=k), scales=scales)
        result = score_pointwise(flagged)
        row = {"k": k, **{key: result[key] for key in
                          ("precision", "recall", "f1", "flagged_pct", "fp")}}
        row["episodes"] = score_episodes(flagged, truth_events)["episodes"]
        if truth_events is not None and len(truth_events):
            events = score_eventwise(flagged, truth_events)
            total = events.loc[events["kind"] == "ALL"]
            row["event_recall"] = float(total["recall"].iloc[0]) if len(total) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
