from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelCreate(BaseModel):
    """Payload for creating a monitored model."""

    name: str
    version: str | None = None


class ModelResponse(BaseModel):
    """Serialized model registry record."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    version: str | None
    created_at: datetime


class ModelListResponse(BaseModel):
    """Paginated response for model listing."""

    items: list[ModelResponse]
    total: int
