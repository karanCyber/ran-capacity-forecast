"""Ingest and clean: 10-minute raw KPI export -> tidy hourly series per cell.

Decisions worth defending in a review:

* Out-of-range values (< 0 or > 100) are collection glitches, not network
  events. They are set to NaN, never clipped into the valid range -- clipping
  would silently invent a plausible-looking reading.
* Short gaps (<= 30 min) are interpolated. Longer gaps are left as NaN and the
  affected hours are marked ``imputed``, so the anomaly detector can exclude
  them. A missing counter is not an anomaly.
* An hour is only aggregated if at least half its sub-hourly samples survived
  cleaning; otherwise the hour is NaN. A "mean" over one sample out of six is
  not a mean.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ran_forecast.config import CONFIG, Config

log = logging.getLogger(__name__)

VALID_MIN, VALID_MAX = 0.0, 100.0


def clean_raw(raw: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Repair the 10-minute grid: drop dupes, void glitches, restore the index."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    before = len(df)
    df = df.drop_duplicates(subset=["cell_id", "timestamp"], keep="last")
    if before != len(df):
        log.info("dropped %d duplicate (cell_id, timestamp) rows", before - len(df))

    out_of_range = (df["prb_util"] < VALID_MIN) | (df["prb_util"] > VALID_MAX)
    log.info("voided %d out-of-range readings (%.3f%%)",
             int(out_of_range.sum()), 100 * out_of_range.mean())
    df.loc[out_of_range, "prb_util"] = np.nan

    freq = f"{cfg.raw_freq_minutes}min"
    full_index = pd.date_range(
        df["timestamp"].min(), df["timestamp"].max(), freq=freq, name="timestamp"
    )

    static_cols = ["site_id", "archetype"]
    static = df.groupby("cell_id", as_index=True)[static_cols].first()

    pieces = []
    for cell_id, group in df.groupby("cell_id", sort=True):
        series = (
            group.set_index("timestamp")[["prb_util", "is_injected"]]
            .reindex(full_index)
        )
        series["is_injected"] = series["is_injected"].fillna(False).astype(bool)
        # Interpolate only short gaps; limit is in units of 10-minute steps.
        max_steps = max(1, 30 // cfg.raw_freq_minutes)
        series["prb_util"] = series["prb_util"].interpolate(
            method="time", limit=max_steps, limit_area="inside"
        )
        series["cell_id"] = cell_id
        pieces.append(series.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    out = out.merge(static, left_on="cell_id", right_index=True, how="left")

    missing = out["prb_util"].isna()
    log.info("after repair: %d rows, %d still missing (%.2f%%)",
             len(out), int(missing.sum()), 100 * missing.mean())
    return out


def to_hourly(clean: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    """Aggregate to hourly means, requiring >= 50% sample coverage per hour."""
    steps_per_hour = 60 // cfg.raw_freq_minutes
    min_samples = max(1, steps_per_hour // 2)

    df = clean.set_index("timestamp")
    grouped = df.groupby("cell_id").resample("1h")

    hourly = grouped.agg(
        prb_util=("prb_util", "mean"),
        n_samples=("prb_util", "count"),
        is_injected=("is_injected", "any"),
    ).reset_index()

    thin = hourly["n_samples"] < min_samples
    hourly.loc[thin, "prb_util"] = np.nan
    log.info("%d hourly buckets below %d-sample threshold -> NaN",
             int(thin.sum()), min_samples)

    # Fill what is left so downstream lag features are contiguous, but keep a
    # flag so imputed hours can be excluded from anomaly scoring.
    hourly["imputed"] = hourly["prb_util"].isna()
    hourly["prb_util"] = (
        hourly.groupby("cell_id")["prb_util"]
        .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
    )

    static = clean.groupby("cell_id")[["site_id", "archetype"]].first()
    hourly = hourly.merge(static, left_on="cell_id", right_index=True, how="left")

    hourly = hourly.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    log.info("hourly dataset: %d rows, %d cells, %s .. %s",
             len(hourly), hourly["cell_id"].nunique(),
             hourly["timestamp"].min(), hourly["timestamp"].max())
    return hourly[
        ["cell_id", "site_id", "archetype", "timestamp", "prb_util",
         "n_samples", "imputed", "is_injected"]
    ]


def data_quality_report(hourly: pd.DataFrame) -> dict:
    return {
        "rows": int(len(hourly)),
        "cells": int(hourly["cell_id"].nunique()),
        "start": str(hourly["timestamp"].min()),
        "end": str(hourly["timestamp"].max()),
        "imputed_pct": round(100 * float(hourly["imputed"].mean()), 3),
        "injected_anomaly_hours_pct": round(100 * float(hourly["is_injected"].mean()), 3),
        "prb_mean": round(float(hourly["prb_util"].mean()), 2),
        "prb_p95": round(float(hourly["prb_util"].quantile(0.95)), 2),
        "prb_max": round(float(hourly["prb_util"].max()), 2),
        "saturated_hours_pct": round(100 * float((hourly["prb_util"] >= 99.0).mean()), 3),
    }


def main() -> None:
    logging.basicConfig(level=CONFIG.log_level, format="%(levelname)s %(name)s: %(message)s")
    raw = pd.read_parquet(CONFIG.raw_path)
    hourly = to_hourly(clean_raw(raw))
    hourly.to_parquet(CONFIG.hourly_path, index=False)
    log.info("wrote %s", CONFIG.hourly_path)
    for key, value in data_quality_report(hourly).items():
        log.info("  %-28s %s", key, value)


if __name__ == "__main__":
    main()
