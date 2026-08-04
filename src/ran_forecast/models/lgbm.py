"""LightGBM day-ahead forecaster.

Trained on ``target_delta = y(t) - y(t-168)``, so a prediction is::

    forecast = seasonal_naive(t) + model(features(t))

then clipped to [0, 100]. See ``features.py`` for why the target is a delta and
why no time-index feature is used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ran_forecast.models.features import (
    BASELINE_COLUMN,
    CATEGORICAL,
    FEATURE_COLUMNS,
    TARGET,
    clip_prb,
)

log = logging.getLogger(__name__)

PARAMS: dict = {
    "objective": "regression_l1",  # L1: robust to the injected anomalies in train
    "metric": "l1",
    "learning_rate": 0.06,
    "num_leaves": 96,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 0,
    "verbosity": -1,
    "seed": 42,
}
NUM_BOOST_ROUND = 500


def _as_matrix(frame: pd.DataFrame, categories: dict[str, list] | None = None) -> pd.DataFrame:
    X = frame[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL:
        if categories is not None:
            X[col] = pd.Categorical(X[col].astype(str), categories=categories[col])
        else:
            X[col] = X[col].astype("category")
    return X


def train(train_frame: pd.DataFrame, num_boost_round: int = NUM_BOOST_ROUND) -> tuple[lgb.Booster, dict]:
    X = _as_matrix(train_frame)
    y = train_frame[TARGET].to_numpy(dtype=float)

    categories = {col: [str(c) for c in X[col].cat.categories] for col in CATEGORICAL}
    dataset = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL, free_raw_data=False)
    booster = lgb.train(PARAMS, dataset, num_boost_round=num_boost_round)

    meta = {
        "categories": categories,
        "feature_columns": FEATURE_COLUMNS,
        "num_boost_round": num_boost_round,
        "params": PARAMS,
        "n_train_rows": int(len(train_frame)),
        "train_end": str(train_frame["timestamp"].max()),
    }
    log.info("trained on %d rows up to %s", len(train_frame), meta["train_end"])
    return booster, meta


def predict(booster: lgb.Booster, frame: pd.DataFrame, meta: dict | None = None) -> np.ndarray:
    """Return absolute PRB forecasts (baseline + learned correction, clipped)."""
    categories = meta["categories"] if meta else None
    X = _as_matrix(frame, categories)
    delta = booster.predict(X, num_iteration=booster.best_iteration or None)
    return clip_prb(frame[BASELINE_COLUMN].to_numpy(dtype=float) + delta)


def feature_importance(booster: lgb.Booster, top: int = 15) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def save(booster: lgb.Booster, meta: dict, model_path: Path, meta_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    meta_path.write_text(json.dumps(meta, indent=2))


def load(model_path: Path, meta_path: Path) -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(model_path))
    meta = json.loads(meta_path.read_text())
    return booster, meta
