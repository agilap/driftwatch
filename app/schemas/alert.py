from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class AlertResponse(BaseModel):
    """Serialized alert row."""

    id: UUID
    model_id: UUID
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
