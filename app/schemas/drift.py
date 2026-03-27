from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel


class DriftScoreResponse(BaseModel):
    """Serialized drift score row for a feature window."""

    id: UUID
    model_id: UUID
    window_date: date
    feature_name: str
    ks_statistic: float | None
    ks_pvalue: float | None
    psi: float | None
    js_divergence: float | None
    weighted_score: float | None
    severity: str | None
    computed_at: datetime


class FeatureDriftSummary(BaseModel):
    """Compact per-feature drift summary."""

    feature_name: str
    psi: float
    severity: str
    weighted_score: float


class ModelDriftSummary(BaseModel):
    """Drift summary for all features in one model/date window."""

    model_id: UUID
    window_date: date
    features: list[FeatureDriftSummary]
    red_count: int
    yellow_count: int
    green_count: int
