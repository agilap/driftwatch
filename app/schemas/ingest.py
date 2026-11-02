from datetime import date, datetime, timedelta
import math
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, field_validator, model_validator

PREDICTIONS_FEATURE_NAME = "__predictions__"


def _validate_features(features: dict[str, list[float]]) -> dict[str, list[float]]:
    if not features:
        raise ValueError("features must not be empty")
    lengths = set()
    for name, values in features.items():
        if name == PREDICTIONS_FEATURE_NAME:
            raise ValueError("use the predictions field for prediction samples")
        if not name.strip() or name != name.strip():
            raise ValueError(
                "feature names must be nonblank without surrounding whitespace"
            )
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"feature '{name}' requires nonempty finite samples")
        lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError("feature sample lengths must match")
    return features


def _validate_importances(values: dict[str, float]) -> dict[str, float]:
    if not values or any(not name.strip() or name != name.strip() for name in values):
        raise ValueError("importances require nonblank feature names")
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("importances must be finite and non-negative")
    if not math.isfinite(sum(values.values())) or sum(values.values()) <= 0:
        raise ValueError("importance total must be finite and positive")
    return values


def _validate_predictions(
    features: dict[str, list[float]], predictions: list[float] | None
) -> None:
    if predictions is not None:
        if len(predictions) != len(next(iter(features.values()))):
            raise ValueError("predictions must match feature sample length")
        if not all(math.isfinite(value) for value in predictions):
            raise ValueError("predictions must be finite")


class ReferencePayload(BaseModel):
    """Payload for registering reference feature distributions."""

    model_id: UUID
    features: dict[str, list[float]]
    predictions: list[float] | None = None
    feature_importances: dict[str, float] | None = None
    importance_method: Literal["shap", "coefficient", "manual"] | None = None

    _features_valid = field_validator("features")(_validate_features)

    @model_validator(mode="after")
    def validate_importances(self):
        _validate_predictions(self.features, self.predictions)
        if self.feature_importances is not None:
            _validate_importances(self.feature_importances)
            if set(self.feature_importances) - set(self.features):
                raise ValueError("importance features must belong to the reference")
        return self


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

    _features_valid = field_validator("features")(_validate_features)

    @model_validator(mode="after")
    def validate_predictions(self):
        _validate_predictions(self.features, self.predictions)
        return self

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


class ImportanceUploadRequest(BaseModel):
    """Payload for uploading feature importances for a model."""

    importances: dict[str, float]
    method: Literal["shap", "coefficient", "manual"]

    _importances_valid = field_validator("importances")(_validate_importances)


class StoredImportanceResponse(BaseModel):
    """Serialized stored feature importance."""

    feature_name: str
    importance: float
    method: str | None


class ImportanceUploadResponse(BaseModel):
    """Response for feature importance upload requests."""

    model_id: UUID
    items: list[StoredImportanceResponse]
    total: int
