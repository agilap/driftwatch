from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


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
