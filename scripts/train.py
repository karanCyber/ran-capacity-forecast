"""Nightly retrain entrypoint.

Run order: generate (optional) -> ingest -> this. In Kubernetes this is what the
CronJob executes; it writes new artifacts to the shared volume that the API
pods read on startup.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ran_forecast.config import CONFIG
from ran_forecast.data import generate, ingest
from ran_forecast.pipeline import build_serving_artifact, write_artifacts

log = logging.getLogger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and refresh serving artifacts")
    parser.add_argument("--regenerate", action="store_true",
                        help="regenerate the synthetic source data first")
    parser.add_argument("--refit-every-days", type=int, default=7,
                        help="backtest refit cadence (1 = refit at every origin)")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="fit and forecast only, no evaluation (faster)")
    args = parser.parse_args()

    logging.basicConfig(
        level=CONFIG.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    CONFIG.artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.regenerate or not CONFIG.raw_path.exists():
        log.info("generating synthetic source data")
        raw, truth = generate.generate(CONFIG)
        raw.to_parquet(CONFIG.raw_path, index=False)
        truth.to_csv(CONFIG.injected_anomalies_path, index=False)

    if args.regenerate or not CONFIG.hourly_path.exists():
        log.info("ingesting and resampling to hourly")
        hourly = ingest.to_hourly(ingest.clean_raw(pd.read_parquet(CONFIG.raw_path)))
        hourly.to_parquet(CONFIG.hourly_path, index=False)

    hourly = pd.read_parquet(CONFIG.hourly_path)
    log.info("loaded %d hourly rows across %d cells", len(hourly), hourly["cell_id"].nunique())

    serving, episodes, metrics = build_serving_artifact(
        hourly, CONFIG, refit_every_days=args.refit_every_days
    )
    write_artifacts(serving, episodes, metrics, CONFIG)

    base = metrics["seasonal_naive"]
    model = metrics["lightgbm"]
    log.info("RESULT baseline MAE %.3f | model MAE %.3f | MASE %.3f | improvement %.1f%%",
             base["mae"], model["mae"], model["mase"], metrics["mae_improvement_pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
