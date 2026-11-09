from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from app.schemas.ingest import MAX_NAME_LENGTH, has_unstorable_text


class ModelCreate(BaseModel):
    """Payload for creating a monitored model."""

    name: str
    version: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model name must not be blank")
        if len(value) > MAX_NAME_LENGTH:
            raise ValueError("model name must be at most 255 characters")
        if not value.isprintable():
            raise ValueError("model name must contain only printable characters")
        if has_unstorable_text(value):
            raise ValueError("model name must not contain NUL or invalid characters")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str | None) -> str | None:
        if value is not None and has_unstorable_text(value):
            raise ValueError("model version must not contain NUL or invalid characters")
        return value


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
