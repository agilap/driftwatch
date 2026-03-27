from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel


class HealthReportResponse(BaseModel):
    """Serialized health report row."""

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
