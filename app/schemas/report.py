from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class HealthReportResponse(BaseModel):
    """Serialized health report row."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    model_id: UUID
    week_start: date
    week_end: date
    overall_score: float | None
    report_json: dict | None
    report_markdown: str | None
    generated_at: datetime


class ReportListResponse(BaseModel):
    """List response for model health reports."""

    items: list[HealthReportResponse]
    total: int


class ReportGenerateRequest(BaseModel):
    """Manual report generation request payload."""

    week_start: date

    @field_validator("week_start")
    @classmethod
    def validate_week_start(cls, value: date) -> date:
        if value > datetime.now(timezone.utc).date():
            raise ValueError("week_start must not be in the future")
        return value
