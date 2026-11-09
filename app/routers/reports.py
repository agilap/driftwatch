from datetime import date, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routers.dependencies import require_model
from app.schemas.report import (
    HealthReportResponse,
    ReportGenerateRequest,
    ReportListResponse,
)
from app.services.report_generator import (
    generate_report,
    get_latest_report_for_model,
    get_report_for_week,
    list_reports_for_model,
)

router = APIRouter(tags=["reports"])


@router.get(
    "/{model_id}",
    response_model=ReportListResponse,
    dependencies=[Depends(require_model)],
)
async def list_reports_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ReportListResponse:
    """List all reports for a model, newest first."""
    rows = await list_reports_for_model(db=db, model_id=model_id)
    items = [HealthReportResponse.model_validate(row) for row in rows]
    return ReportListResponse(items=items, total=len(items))


@router.get(
    "/{model_id}/latest",
    response_model=HealthReportResponse,
    dependencies=[Depends(require_model)],
)
async def latest_report_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> HealthReportResponse:
    """Fetch the latest report for a model."""
    row = await get_latest_report_for_model(db=db, model_id=model_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No reports found for model")
    return HealthReportResponse.model_validate(row)


@router.get(
    "/{model_id}/{week_start}",
    response_model=HealthReportResponse,
    dependencies=[Depends(require_model)],
)
async def get_report_week_endpoint(
    model_id: UUID,
    week_start: date,
    db: AsyncSession = Depends(get_db),
) -> HealthReportResponse:
    """Fetch a report for a specific week start date."""
    row = await get_report_for_week(db=db, model_id=model_id, week_start=week_start)
    if row is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return HealthReportResponse.model_validate(row)


@router.post(
    "/{model_id}/generate",
    response_model=HealthReportResponse,
    dependencies=[Depends(require_model)],
)
async def generate_report_endpoint(
    model_id: UUID,
    payload: ReportGenerateRequest,
    db: AsyncSession = Depends(get_db),
) -> HealthReportResponse:
    """Manually generate a weekly report for a model."""
    week_start = payload.week_start
    week_end = week_start + timedelta(days=6)
    report = await generate_report(
        db=db, model_id=model_id, week_start=week_start, week_end=week_end
    )
    return HealthReportResponse.model_validate(report)
