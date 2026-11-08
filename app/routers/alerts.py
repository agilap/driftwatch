from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routers.dependencies import require_model
from app.schemas.alert import AlertListResponse, AlertResponse
from app.services.alert_service import count_alerts, list_alerts, resolve_alert

router = APIRouter(tags=["alerts"])


async def _list_alerts_response(
    db: AsyncSession, model_id: UUID | None, resolved: bool, page: int, page_size: int
) -> AlertListResponse:
    rows = await list_alerts(
        db=db, model_id=model_id, resolved=resolved, page=page, page_size=page_size
    )
    total = await count_alerts(db=db, model_id=model_id, resolved=resolved)
    return AlertListResponse(
        items=[AlertResponse.model_validate(row) for row in rows], total=total
    )


@router.get("", response_model=AlertListResponse)
async def list_active_alerts_endpoint(
    model_id: UUID | None = Query(None),
    resolved: bool = Query(False),
    page: int = Query(1, ge=1, le=1_000_000),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> AlertListResponse:
    """List alerts, unresolved by default, with optional model filtering."""
    return await _list_alerts_response(db, model_id, resolved, page, page_size)


@router.get(
    "/{model_id}",
    response_model=AlertListResponse,
    dependencies=[Depends(require_model)],
)
async def list_model_alerts_endpoint(
    model_id: UUID,
    resolved: bool = Query(False),
    page: int = Query(1, ge=1, le=1_000_000),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> AlertListResponse:
    """List alerts for a specific model."""
    return await _list_alerts_response(db, model_id, resolved, page, page_size)


@router.patch("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert_endpoint(
    alert_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AlertResponse:
    """Mark an alert as resolved."""
    row = await resolve_alert(db=db, alert_id=alert_id)
    return AlertResponse.model_validate(row)
