from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AlertResponse(BaseModel):
    """Serialized alert row."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_id: UUID
    window_date: date | None
    feature_name: str | None
    alert_type: str | None
    severity: str | None
    message: str | None
    resolved_at: datetime | None
    created_at: datetime


class AlertListResponse(BaseModel):
    """Paginated alert list response."""

    items: list[AlertResponse]
    total: int
