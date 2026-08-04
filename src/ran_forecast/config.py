"""Central configuration.

Every value can be overridden by an environment variable, which is how the
Kubernetes ConfigMap drives the container without rebuilding the image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Config:
    # ---- storage -------------------------------------------------------
    artifact_dir: Path = field(
        default_factory=lambda: _env_path("ARTIFACT_DIR", REPO_ROOT / "artifacts")
    )

    # ---- synthetic data generation -------------------------------------
    n_cells: int = field(default_factory=lambda: _env_int("N_CELLS", 60))
    n_days: int = field(default_factory=lambda: _env_int("N_DAYS", 120))
    raw_freq_minutes: int = field(default_factory=lambda: _env_int("RAW_FREQ_MINUTES", 10))
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))

    # ---- forecasting ---------------------------------------------------
    horizon_hours: int = field(default_factory=lambda: _env_int("HORIZON_HOURS", 24))
    backtest_days: int = field(default_factory=lambda: _env_int("BACKTEST_DAYS", 21))

    # ---- anomaly detection ---------------------------------------------
    anomaly_k: float = field(default_factory=lambda: _env_float("ANOMALY_K", 3.5))
    anomaly_consecutive: int = field(default_factory=lambda: _env_int("ANOMALY_CONSECUTIVE", 2))

    # ---- api -----------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ---- derived paths -------------------------------------------------
    @property
    def raw_path(self) -> Path:
        return self.artifact_dir / "raw_10min.parquet"

    @property
    def hourly_path(self) -> Path:
        return self.artifact_dir / "hourly.parquet"

    @property
    def injected_anomalies_path(self) -> Path:
        return self.artifact_dir / "injected_anomalies.csv"

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "model.txt"

    @property
    def model_meta_path(self) -> Path:
        return self.artifact_dir / "model_meta.json"

    @property
    def forecasts_path(self) -> Path:
        return self.artifact_dir / "forecasts.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.artifact_dir / "metrics.json"


CONFIG = Config()

# The seasonal period we care about: same hour, one week back.
WEEKLY_LAG_HOURS = 168
DAILY_LAG_HOURS = 24
