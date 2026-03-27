from datetime import date, datetime, timedelta
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, field_validator


class ReferencePayload(BaseModel):
    """Payload for registering reference feature distributions."""

    model_id: UUID
    features: dict[str, list[float]]
    feature_importances: dict[str, float] | None = None
    importance_method: Literal["shap", "coefficient", "manual"] | None = None


class ReferenceResponse(BaseModel):
    """Response returned after reference registration."""

    model_id: UUID
    features_registered: list[str]
    importances_registered: bool
    registered_at: datetime


class RegisteredFeatureResponse(BaseModel):
    """Serialized reference feature entry."""

    feature_name: str
    stats: dict[str, float]


class ReferenceListResponse(BaseModel):
    """List of registered reference features for a model."""

    model_id: UUID
    items: list[RegisteredFeatureResponse]
    total: int


class SnapshotPayload(BaseModel):
    """Payload for ingesting production feature snapshots."""

    model_id: UUID
    timestamp: datetime
    features: dict[str, list[float]]
    predictions: list[float] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_utc(cls, value: datetime) -> datetime:
        """Require timezone-aware UTC timestamps."""
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            msg = "timestamp must be UTC"
            raise ValueError(msg)
        return value


class SnapshotResponse(BaseModel):
    """Response returned after snapshot ingestion."""

    model_id: UUID
    window_date: date
    features_recorded: list[str]
    sample_count: int
    drift_triggered: bool
    recorded_at: datetime


class SnapshotFeatureResponse(BaseModel):
    """Serialized snapshot feature record for a single date window."""

    feature_name: str
    stats: dict[str, float]
    distribution: dict[str, list[float] | list[int]]
    sample_count: int | None
