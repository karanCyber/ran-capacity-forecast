"""FastAPI service.

Design: the app serves *precomputed* artifacts loaded into memory at startup.
No training, no inference, no file reads on the request path. Consequences:

* p99 latency is a dictionary lookup, not a model forward pass
* pods scale horizontally with no shared state and no GPU
* a bad model is rolled back by swapping a parquet file and restarting

``/readyz`` fails until the artifacts are loaded, so Kubernetes will not send
traffic to a pod whose volume has not mounted yet. ``/healthz`` only reports
that the process is alive -- conflating the two is how you get a liveness probe
that restarts every pod in the deployment the moment a shared volume hiccups.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from ran_forecast import __version__
from ran_forecast.anomaly import CAPACITY_RISK_THRESHOLD
from ran_forecast.api.dashboard import DASHBOARD_HTML
from ran_forecast.api.schemas import (
    AnomalyEpisode,
    AnomalyResponse,
    CapacityRecommendation,
    ForecastPoint,
    ForecastResponse,
    HealthResponse,
    ReadyResponse,
)
from ran_forecast.config import CONFIG

log = logging.getLogger("api")

SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


class Store:
    """In-memory artifact store, populated once at startup."""

    def __init__(self) -> None:
        self.forecasts: pd.DataFrame | None = None
        self.episodes: pd.DataFrame | None = None
        self.metrics: dict = {}
        self.loaded_at: datetime | None = None

    @property
    def ready(self) -> bool:
        return self.forecasts is not None and not self.forecasts.empty

    def load(self) -> None:
        if not CONFIG.forecasts_path.exists():
            log.warning("artifact %s not found -- service will report not ready",
                        CONFIG.forecasts_path)
            return

        forecasts = pd.read_parquet(CONFIG.forecasts_path)
        forecasts["timestamp"] = pd.to_datetime(forecasts["timestamp"], utc=True)
        self.forecasts = forecasts.sort_values(["cell_id", "timestamp"])

        episodes_path = CONFIG.artifact_dir / "episodes.parquet"
        if episodes_path.exists():
            episodes = pd.read_parquet(episodes_path)
            for col in ("start", "end"):
                episodes[col] = pd.to_datetime(episodes[col], utc=True)
            self.episodes = episodes
        else:
            self.episodes = pd.DataFrame()

        if CONFIG.metrics_path.exists():
            self.metrics = json.loads(CONFIG.metrics_path.read_text())

        self.loaded_at = datetime.now(timezone.utc)
        log.info("loaded %d forecast rows, %d anomaly episodes, %d cells",
                 len(self.forecasts), len(self.episodes),
                 self.forecasts["cell_id"].nunique())


STORE = Store()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=CONFIG.log_level, format="%(levelname)s %(name)s: %(message)s")
    STORE.load()
    yield


app = FastAPI(
    title="RAN Capacity Forecast",
    version=__version__,
    description=(
        "Cell-level PRB utilisation forecasting, anomaly detection and capacity "
        "recommendations. Conceptually a non-RT RIC rApp: consumes cell KPIs, "
        "emits capacity guidance on a >1s control loop."
    ),
    lifespan=lifespan,
)


def _require_ready() -> pd.DataFrame:
    if not STORE.ready:
        raise HTTPException(
            status_code=503,
            detail="Forecast artifacts not loaded. Run `make train` or wait for the "
                   "retrain CronJob to populate the shared volume.",
        )
    return STORE.forecasts


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    """Liveness: is the process up. Deliberately does not check artifacts."""
    return HealthResponse(status="ok", version=__version__)


@app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
def readyz() -> ReadyResponse:
    """Readiness: can this pod actually serve requests."""
    if not STORE.ready:
        raise HTTPException(status_code=503, detail="artifacts not loaded")
    meta = STORE.metrics.get("model_meta", {})
    return ReadyResponse(
        status="ready",
        cells=int(STORE.forecasts["cell_id"].nunique()),
        rows=int(len(STORE.forecasts)),
        artifact_generated_at=STORE.loaded_at,
        model_trained_at=meta.get("trained_at"),
    )


@app.get("/cells", tags=["catalogue"])
def cells() -> dict:
    """List known cells, so the API is explorable without guessing IDs."""
    df = _require_ready()
    catalogue = (
        df.groupby("cell_id")
        .agg(site_id=("site_id", "first"), archetype=("archetype", "first"))
        .reset_index()
    )
    return {"count": len(catalogue), "cells": catalogue.to_dict(orient="records")}


@app.get("/forecast/{cell_id}", response_model=ForecastResponse, tags=["forecast"])
def forecast(
    cell_id: str,
    horizon: int = Query(24, ge=1, le=24, description="Hours ahead to return"),
    include_history: int = Query(
        0, ge=0, le=336, description="Hours of scored history to include alongside"
    ),
) -> ForecastResponse:
    """Day-ahead PRB utilisation forecast for one cell.

    The seasonal-naive baseline is returned alongside every point on purpose:
    the comparison that justifies the model should be visible in the API, not
    only in the README.
    """
    df = _require_ready()
    cell = df[df["cell_id"] == cell_id]
    if cell.empty:
        raise HTTPException(status_code=404, detail=f"Unknown cell_id: {cell_id}")

    future = cell[cell["is_forecast"]].nsmallest(horizon, "timestamp")
    frames = [future]
    if include_history:
        history = cell[~cell["is_forecast"]].nlargest(include_history, "timestamp")
        frames.insert(0, history)
    window = pd.concat(frames).sort_values("timestamp")

    points = [
        ForecastPoint(
            timestamp=row.timestamp,
            yhat=round(float(row.yhat_model), 3),
            yhat_baseline=None if pd.isna(row.yhat_baseline) else round(float(row.yhat_baseline), 3),
            actual=None if pd.isna(row.prb_util) else round(float(row.prb_util), 3),
            is_forecast=bool(row.is_forecast),
            capacity_risk=bool(row.capacity_risk),
        )
        for row in window.itertuples(index=False)
    ]

    peak = float(future["yhat_model"].max()) if not future.empty else float("nan")
    return ForecastResponse(
        cell_id=cell_id,
        site_id=cell["site_id"].iloc[0],
        archetype=cell["archetype"].iloc[0],
        horizon_hours=horizon,
        generated_at=STORE.loaded_at or datetime.now(timezone.utc),
        capacity_risk_threshold=CAPACITY_RISK_THRESHOLD,
        peak_forecast=round(peak, 3),
        hours_at_risk=int(future["capacity_risk"].sum()),
        points=points,
    )


@app.get("/anomalies", response_model=AnomalyResponse, tags=["anomaly"])
def anomalies(
    since: datetime | None = Query(None, description="Only episodes ending at or after this time"),
    cell_id: str | None = Query(None, description="Restrict to one cell"),
    severity: str | None = Query(None, pattern="^(low|medium|high)$"),
    limit: int = Query(50, ge=1, le=500),
) -> AnomalyResponse:
    """Detected anomaly episodes, most severe first.

    Episodes, not individual hours: consecutive flagged hours are collapsed into
    one alert, because that is the unit an operations team acts on.
    """
    _require_ready()
    episodes = STORE.episodes
    if episodes is None or episodes.empty:
        return AnomalyResponse(count=0, since=since, severity_filter=severity, episodes=[])

    view = episodes.copy()
    if since is not None:
        cutoff = pd.Timestamp(since)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        view = view[view["end"] >= cutoff]
    if cell_id:
        view = view[view["cell_id"] == cell_id]
    if severity:
        minimum = SEVERITY_ORDER[severity]
        view = view[view["severity"].map(SEVERITY_ORDER).fillna(0) >= minimum]

    view = view.reindex(view["peak_z"].abs().sort_values(ascending=False).index).head(limit)

    return AnomalyResponse(
        count=int(len(view)),
        since=since,
        severity_filter=severity,
        episodes=[
            AnomalyEpisode(
                cell_id=row.cell_id,
                start=row.start,
                end=row.end,
                hours=int(row.hours),
                peak_z=round(float(row.peak_z), 2),
                severity=str(row.severity),
                direction=str(row.direction),
                mean_actual=round(float(row.mean_actual), 2),
                mean_forecast=round(float(row.mean_forecast), 2),
            )
            for row in view.itertuples(index=False)
        ],
    )


@app.get("/capacity-risk", response_model=list[CapacityRecommendation], tags=["forecast"])
def capacity_risk(limit: int = Query(20, ge=1, le=200)) -> list[CapacityRecommendation]:
    """Cells forecast to breach the utilisation threshold in the next 24h.

    This is the actual *recommendation* output -- the part a planning team would
    consume. Forecasting is the means, not the deliverable.
    """
    df = _require_ready()
    future = df[df["is_forecast"]]
    if future.empty:
        return []

    grouped = (
        future.groupby(["cell_id", "site_id", "archetype"], observed=True)
        .agg(
            peak_forecast=("yhat_model", "max"),
            hours_at_risk=("capacity_risk", "sum"),
            first_risk_hour=("timestamp", lambda s: s.min()),
        )
        .reset_index()
    )
    at_risk = future[future["capacity_risk"]].groupby("cell_id")["timestamp"].min()
    grouped["first_risk_hour"] = grouped["cell_id"].map(at_risk)
    grouped = grouped[grouped["hours_at_risk"] > 0].sort_values(
        ["hours_at_risk", "peak_forecast"], ascending=False
    ).head(limit)

    out = []
    for row in grouped.itertuples(index=False):
        if row.hours_at_risk >= 6:
            action = "Sustained congestion forecast: schedule carrier addition or cell split."
        elif row.hours_at_risk >= 3:
            action = "Repeated busy-hour breach: review sector load balancing."
        else:
            action = "Isolated breach: monitor, no action yet."
        out.append(
            CapacityRecommendation(
                cell_id=row.cell_id,
                site_id=row.site_id,
                archetype=row.archetype,
                peak_forecast=round(float(row.peak_forecast), 2),
                hours_at_risk=int(row.hours_at_risk),
                first_risk_hour=row.first_risk_hour,
                recommendation=action,
            )
        )
    return out


@app.get("/metrics-summary", tags=["ops"])
def metrics_summary() -> JSONResponse:
    """The results table, served. Keeps the model's claims auditable at runtime."""
    if not STORE.metrics:
        raise HTTPException(status_code=503, detail="metrics not available")
    keys = (
        "seasonal_naive", "seasonal_naive_cleaned", "lightgbm",
        "mae_improvement_pct", "eval_start", "eval_end", "horizon_hours",
    )
    summary = {key: STORE.metrics.get(key) for key in keys if key in STORE.metrics}
    if "anomaly" in STORE.metrics:
        summary["anomaly"] = {
            "pointwise": STORE.metrics["anomaly"].get("pointwise"),
            "episodes": STORE.metrics["anomaly"].get("episodes"),
        }
    return JSONResponse(summary)


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    """Single self-contained page. Renders its own SVG; no CDN, no build step."""
    return DASHBOARD_HTML


@app.get("/", response_class=PlainTextResponse, include_in_schema=False)
def root() -> str:
    return (
        "ran-capacity-forecast\n"
        "  GET /forecast/{cell_id}?horizon=24&include_history=48\n"
        "  GET /anomalies?since=...&severity=high\n"
        "  GET /capacity-risk\n"
        "  GET /cells   /metrics-summary   /healthz   /readyz\n"
        "  GET /dashboard\n"
        "  GET /docs\n"
    )
