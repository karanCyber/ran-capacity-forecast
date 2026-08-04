"""Response models. Explicit schemas keep /docs useful and the contract stable."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    timestamp: datetime
    yhat: float = Field(..., description="LightGBM forecast, PRB utilisation %")
    yhat_baseline: float | None = Field(
        None, description="Seasonal-naive forecast (same hour last week) for comparison"
    )
    actual: float | None = Field(None, description="Observed value, null for future hours")
    is_forecast: bool = Field(..., description="True if this hour is in the future")
    capacity_risk: bool = Field(..., description="Forecast at or above the risk threshold")


class ForecastResponse(BaseModel):
    cell_id: str
    site_id: str | None = None
    archetype: str | None = None
    horizon_hours: int
    generated_at: datetime
    capacity_risk_threshold: float
    peak_forecast: float
    hours_at_risk: int
    points: list[ForecastPoint]


class AnomalyEpisode(BaseModel):
    cell_id: str
    start: datetime
    end: datetime
    hours: int
    peak_z: float
    severity: str
    direction: str
    mean_actual: float | None = None
    mean_forecast: float | None = None


class AnomalyResponse(BaseModel):
    count: int
    since: datetime | None = None
    severity_filter: str | None = None
    episodes: list[AnomalyEpisode]


class CapacityRecommendation(BaseModel):
    cell_id: str
    site_id: str | None = None
    archetype: str | None = None
    peak_forecast: float
    hours_at_risk: int
    first_risk_hour: datetime | None = None
    recommendation: str


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    cells: int
    rows: int
    artifact_generated_at: datetime | None = None
    model_trained_at: str | None = None
