from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.alert import AlertListResponse, AlertResponse
from app.services.alert_service import count_alerts, list_alerts, resolve_alert
from app.services.model_service import get_model

router = APIRouter(tags=["alerts"])


@router.get("", response_model=AlertListResponse)
async def list_active_alerts_endpoint(
    model_id: UUID | None = Query(None),
    resolved: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> AlertListResponse:
    """List alerts, unresolved by default, with optional model filtering."""
    rows = await list_alerts(
        db=db,
        model_id=model_id,
        resolved=resolved,
        page=page,
        page_size=page_size,
    )
    items = [
        AlertResponse(
            id=row.id,
            model_id=row.model_id,
            feature_name=row.feature_name,
            alert_type=row.alert_type,
            severity=row.severity,
            message=row.message,
            resolved_at=row.resolved_at,
            created_at=row.created_at,
        )
        for row in rows
    ]
    total = await count_alerts(db=db, model_id=model_id, resolved=resolved)
    return AlertListResponse(items=items, total=total)


@router.get("/{model_id}", response_model=AlertListResponse)
async def list_model_alerts_endpoint(
    model_id: UUID,
    resolved: bool = Query(False),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> AlertListResponse:
    """List alerts for a specific model."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    rows = await list_alerts(
        db=db,
        model_id=model_id,
        resolved=resolved,
        page=page,
        page_size=page_size,
    )
    items = [
        AlertResponse(
            id=row.id,
            model_id=row.model_id,
            feature_name=row.feature_name,
            alert_type=row.alert_type,
            severity=row.severity,
            message=row.message,
            resolved_at=row.resolved_at,
            created_at=row.created_at,
        )
        for row in rows
    ]
    total = await count_alerts(db=db, model_id=model_id, resolved=resolved)
    return AlertListResponse(items=items, total=total)


@router.patch("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert_endpoint(
    alert_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AlertResponse:
    """Mark an alert as resolved."""
    row = await resolve_alert(db=db, alert_id=alert_id)
    return AlertResponse(
        id=row.id,
        model_id=row.model_id,
        feature_name=row.feature_name,
        alert_type=row.alert_type,
        severity=row.severity,
        message=row.message,
        resolved_at=row.resolved_at,
        created_at=row.created_at,
    )
