"""Synthetic PRB-utilisation generator.

This is NOT real operator data. It is a deliberately transparent simulation of
what a cell-level PRB (Physical Resource Block) utilisation KPI looks like, so
that the forecasting and anomaly-detection pipeline can be evaluated against
known ground truth.

Model
-----
Utilisation for cell ``c`` at time ``t`` is multiplicative::

    prb(t) = base_c * daily_a(t) * weekly_a(t) * growth_c(t) * noise_c(t)

then clipped to [0, 100] because PRB utilisation is a bounded percentage.

* ``a`` is the cell *archetype* (residential / business / transit / mixed),
  which sets the shape of the daily and weekly profiles.
* ``growth`` is a linear trend, standing in for a network in active rollout
  where offered traffic climbs month on month.
* ``noise`` is AR(1) in log space, so residuals are autocorrelated rather than
  white. That is realistic and it is what makes the "two consecutive breaches"
  rule in the anomaly detector do real work.

Anomalies are injected on purpose and written to a ground-truth CSV, which lets
us score the detector with precision/recall instead of eyeballing a chart.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ran_forecast.config import CONFIG, Config

log = logging.getLogger(__name__)

ARCHETYPES = ("residential", "business", "transit", "mixed")

# Relative load at each hour of the day (index 0..23). Normalised to mean 1.0
# inside _normalise(). These are hand-drawn shapes, not fitted to anything.
_DAILY_KNOTS: dict[str, list[float]] = {
    # Quiet overnight, dip while people are at work, strong evening peak.
    "residential": [
        0.30, 0.22, 0.18, 0.16, 0.17, 0.25, 0.45, 0.70,
        0.85, 0.80, 0.72, 0.70, 0.72, 0.70, 0.68, 0.72,
        0.85, 1.05, 1.30, 1.50, 1.55, 1.45, 1.10, 0.60,
    ],
    # Office hours. Nearly dead overnight.
    "business": [
        0.12, 0.10, 0.09, 0.09, 0.10, 0.15, 0.30, 0.65,
        1.20, 1.55, 1.65, 1.60, 1.45, 1.55, 1.60, 1.55,
        1.35, 1.00, 0.60, 0.38, 0.28, 0.22, 0.18, 0.14,
    ],
    # Twin commute peaks.
    "transit": [
        0.20, 0.14, 0.11, 0.10, 0.14, 0.35, 0.85, 1.60,
        1.75, 1.20, 0.85, 0.80, 0.85, 0.85, 0.85, 1.00,
        1.35, 1.70, 1.60, 1.10, 0.80, 0.60, 0.42, 0.28,
    ],
    # Flatter, mixed land use.
    "mixed": [
        0.35, 0.28, 0.24, 0.22, 0.24, 0.34, 0.55, 0.85,
        1.10, 1.15, 1.10, 1.12, 1.15, 1.12, 1.10, 1.15,
        1.25, 1.35, 1.40, 1.35, 1.20, 1.00, 0.75, 0.50,
    ],
}

# Relative load by day of week (Monday=0 .. Sunday=6).
_WEEKLY: dict[str, list[float]] = {
    "residential": [0.95, 0.95, 0.96, 0.98, 1.05, 1.12, 1.08],
    "business":    [1.05, 1.08, 1.08, 1.06, 0.95, 0.40, 0.28],
    "transit":     [1.05, 1.06, 1.06, 1.05, 1.10, 0.62, 0.45],
    "mixed":       [1.00, 1.01, 1.01, 1.00, 1.05, 0.90, 0.82],
}


def _normalise(values: np.ndarray) -> np.ndarray:
    return values / values.mean()


def _daily_profile_at(archetype: str, minute_of_day: np.ndarray) -> np.ndarray:
    """Interpolate the 24 hourly knots smoothly onto sub-hourly timestamps.

    Wraps around midnight so there is no discontinuity at 00:00.
    """
    knots = _normalise(np.asarray(_DAILY_KNOTS[archetype], dtype=float))
    # Knot at the centre of each hour, plus wrapped endpoints for periodicity.
    x = np.concatenate([[-30.0], np.arange(24) * 60 + 30, [24 * 60 + 30]])
    y = np.concatenate([[knots[-1]], knots, [knots[0]]])
    return np.interp(minute_of_day, x, y)


@dataclass
class InjectedAnomaly:
    cell_id: str
    start: pd.Timestamp
    end: pd.Timestamp
    kind: str
    magnitude: float


def _make_cell_table(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """One row per cell: archetype, base load, growth rate, site grouping."""
    n = cfg.n_cells
    archetype = rng.choice(ARCHETYPES, size=n, p=[0.35, 0.25, 0.20, 0.20])

    # Mean utilisation the cell sits at. A handful are deliberately hot so that
    # clipping at 100% actually bites and the busy-hour saturation is visible.
    base = rng.uniform(22.0, 52.0, size=n)
    hot = rng.random(n) < 0.15
    base[hot] = rng.uniform(58.0, 74.0, size=hot.sum())

    # Monthly growth, i.e. active rollout. Some cells are flat, none shrink.
    growth_monthly = rng.uniform(0.004, 0.020, size=n)

    return pd.DataFrame(
        {
            "cell_id": [f"CELL_{i:04d}" for i in range(n)],
            "archetype": archetype,
            "base_prb": base,
            "growth_monthly": growth_monthly,
            "site_id": [f"SITE_{i // 3:04d}" for i in range(n)],
        }
    )


def _ar1_noise(n: int, rng: np.random.Generator, phi: float, sigma: float) -> np.ndarray:
    """AR(1) process in log space -> autocorrelated multiplicative noise."""
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(1 - phi**2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return np.exp(out)


def _inject_anomalies(
    df: pd.DataFrame,
    cells: pd.DataFrame,
    rng: np.random.Generator,
    steps_per_hour: int,
) -> tuple[pd.DataFrame, list[InjectedAnomaly]]:
    """Overwrite slices of the series with outages, spikes and drifts.

    Only the last 60% of the timeline is eligible, so the training window stays
    mostly clean and the evaluation window is where detection is scored.
    """
    records: list[InjectedAnomaly] = []
    timestamps = df.index.get_level_values("timestamp").unique().sort_values()
    eligible_start = timestamps[int(len(timestamps) * 0.40)]
    eligible = timestamps[timestamps >= eligible_start]

    values = df["prb_util"].copy()

    for cell_id in cells["cell_id"]:
        n_events = rng.poisson(2.2)
        for _ in range(n_events):
            kind = rng.choice(["outage", "spike", "drift"], p=[0.35, 0.45, 0.20])

            if kind == "outage":
                dur_h = rng.integers(2, 9)
                magnitude = rng.uniform(0.0, 0.08)
            elif kind == "spike":
                dur_h = rng.integers(3, 8)
                magnitude = rng.uniform(1.6, 2.6)
            else:  # drift: slow congestion build-up over days
                dur_h = int(rng.integers(48, 121))
                magnitude = rng.uniform(1.25, 1.6)

            n_steps = dur_h * steps_per_hour
            if n_steps >= len(eligible):
                continue
            start_idx = int(rng.integers(0, len(eligible) - n_steps))
            start = eligible[start_idx]
            end = eligible[start_idx + n_steps - 1]

            key = (cell_id, slice(start, end))
            try:
                segment = values.loc[key]
            except KeyError:
                continue
            if segment.empty:
                continue

            if kind == "drift":
                # Ramp linearly from 1.0 up to `magnitude` and back down.
                ramp = np.linspace(0.0, np.pi, len(segment))
                factor = 1.0 + (magnitude - 1.0) * np.sin(ramp)
            else:
                factor = np.full(len(segment), magnitude)

            values.loc[key] = segment.to_numpy() * factor
            records.append(
                InjectedAnomaly(cell_id, pd.Timestamp(start), pd.Timestamp(end), kind, float(magnitude))
            )

    df = df.copy()
    df["prb_util"] = values.clip(0.0, 100.0)
    df["is_injected"] = False
    for rec in records:
        df.loc[(rec.cell_id, slice(rec.start, rec.end)), "is_injected"] = True
    return df, records


def _corrupt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the mess a real KPI export has: gaps and out-of-range glitches.

    The ingest stage is responsible for repairing this. Corrupted rows are NOT
    labelled as anomalies -- they are collection faults, not network events, and
    conflating the two is exactly the mistake this pipeline should avoid.
    """
    df = df.reset_index()

    # Out-of-range sensor glitches: impossible percentages.
    glitch_mask = rng.random(len(df)) < 0.0008
    glitch_values = rng.choice([-1.0, 999.0, 0.0], size=int(glitch_mask.sum()))
    df.loc[glitch_mask, "prb_util"] = glitch_values

    # Whole rows missing from the export, in contiguous blocks per cell.
    drop_mask = np.zeros(len(df), dtype=bool)
    for cell_id, group in df.groupby("cell_id", sort=False):
        if rng.random() < 0.30:  # 30% of cells have at least one gap
            for _ in range(int(rng.integers(1, 4))):
                gap_len = int(rng.integers(3, 40))  # 30 min .. ~6.5 h
                if gap_len >= len(group):
                    continue
                start = int(rng.integers(0, len(group) - gap_len))
                drop_mask[group.index[start : start + gap_len]] = True

    df = df.loc[~drop_mask].reset_index(drop=True)
    return df


def generate(cfg: Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the raw 10-minute dataset and the injected-anomaly ground truth."""
    rng = np.random.default_rng(cfg.random_seed)
    cells = _make_cell_table(cfg, rng)

    freq = f"{cfg.raw_freq_minutes}min"
    steps_per_hour = 60 // cfg.raw_freq_minutes
    end = pd.Timestamp("2025-06-30 23:50", tz="UTC").floor(freq)
    index = pd.date_range(end=end, periods=cfg.n_days * 24 * steps_per_hour, freq=freq, name="timestamp")

    minute_of_day = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    dow = index.dayofweek.to_numpy()
    # Months elapsed since the start of the window, for the growth term.
    months = (index - index[0]).total_seconds().to_numpy() / (30.0 * 86400.0)

    frames = []
    for row in cells.itertuples(index=False):
        daily = _daily_profile_at(row.archetype, minute_of_day)
        weekly = _normalise(np.asarray(_WEEKLY[row.archetype], dtype=float))[dow]
        growth = 1.0 + row.growth_monthly * months
        noise = _ar1_noise(len(index), rng, phi=0.65, sigma=0.055)

        prb = row.base_prb * daily * weekly * growth * noise
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "cell_id": row.cell_id,
                    "site_id": row.site_id,
                    "archetype": row.archetype,
                    "prb_util": np.clip(prb, 0.0, 100.0),
                }
            )
        )

    df = pd.concat(frames, ignore_index=True).set_index(["cell_id", "timestamp"]).sort_index()

    df, records = _inject_anomalies(df, cells, rng, steps_per_hour)
    truth = pd.DataFrame(
        [
            {
                "cell_id": r.cell_id,
                "start": r.start,
                "end": r.end,
                "kind": r.kind,
                "magnitude": round(r.magnitude, 3),
            }
            for r in records
        ]
    ).sort_values(["cell_id", "start"])

    raw = _corrupt(df, rng)
    log.info("generated %d rows across %d cells, %d injected anomaly events",
             len(raw), cfg.n_cells, len(truth))
    return raw, truth


def main() -> None:
    logging.basicConfig(level=CONFIG.log_level, format="%(levelname)s %(name)s: %(message)s")
    CONFIG.artifact_dir.mkdir(parents=True, exist_ok=True)
    raw, truth = generate(CONFIG)
    raw.to_parquet(CONFIG.raw_path, index=False)
    truth.to_csv(CONFIG.injected_anomalies_path, index=False)
    log.info("wrote %s (%d rows)", CONFIG.raw_path, len(raw))
    log.info("wrote %s (%d events)", CONFIG.injected_anomalies_path, len(truth))


if __name__ == "__main__":
    main()
