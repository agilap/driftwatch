from datetime import date
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.ingest import (
    ImportanceUploadRequest,
    ImportanceUploadResponse,
    ReferenceListResponse,
    ReferencePayload,
    ReferenceResponse,
    SnapshotFeatureResponse,
    SnapshotPayload,
    SnapshotResponse,
    StoredImportanceResponse,
)
from app.services.drift_engine import run_drift_analysis
from app.services import importance_scorer
from app.services.model_service import get_model
from app.services.reference_service import list_reference_features, register_reference
from app.services.snapshot_service import get_snapshot_window, ingest_snapshot, list_snapshot_dates

router = APIRouter(tags=["ingest"])
model_importance_router = APIRouter(tags=["ingest"])


@router.post("/reference", response_model=ReferenceResponse)
async def register_reference_endpoint(
    payload: ReferencePayload,
    db: AsyncSession = Depends(get_db),
) -> ReferenceResponse:
    """Register reference distributions and optional feature importances."""
    model = await get_model(db=db, model_id=payload.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return await register_reference(db=db, payload=payload)


@router.get("/reference/{model_id}", response_model=ReferenceListResponse)
async def list_reference_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ReferenceListResponse:
    """List all registered reference features for a model."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return await list_reference_features(db=db, model_id=model_id)


@router.post("/snapshot", response_model=SnapshotResponse)
async def ingest_snapshot_endpoint(
    payload: SnapshotPayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> SnapshotResponse:
    """Ingest production snapshots and queue drift analysis asynchronously."""
    response = await ingest_snapshot(db=db, payload=payload)
    background_tasks.add_task(run_drift_analysis, db, response.model_id, response.window_date)
    return response


@router.get("/snapshots/{model_id}", response_model=list[date])
async def list_snapshot_dates_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[date]:
    """List all recorded snapshot windows for a model."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return await list_snapshot_dates(db=db, model_id=model_id)


@router.get("/snapshots/{model_id}/{window_date}", response_model=list[SnapshotFeatureResponse])
async def get_snapshot_window_endpoint(
    model_id: UUID,
    window_date: date,
    db: AsyncSession = Depends(get_db),
) -> list[SnapshotFeatureResponse]:
    """Get per-feature snapshot stats for a specific model and date window."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    rows = await get_snapshot_window(db=db, model_id=model_id, window_date=window_date)
    return [
        SnapshotFeatureResponse(
            feature_name=row.feature_name,
            stats=row.stats,
            distribution=row.distribution,
            sample_count=row.sample_count,
        )
        for row in rows
    ]


@model_importance_router.post("/models/{model_id}/importances", response_model=ImportanceUploadResponse)
async def upload_model_importances_endpoint(
    model_id: UUID,
    payload: ImportanceUploadRequest,
    db: AsyncSession = Depends(get_db),
) -> ImportanceUploadResponse:
    """Upload and upsert model feature importances."""
    model = await get_model(db=db, model_id=model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        normalized = importance_scorer.import_from_json(
            json_data=payload.importances,
            method=payload.method,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rows = await importance_scorer.upsert_importances(
        db=db,
        model_id=model_id,
        importances=normalized,
        method=payload.method,
    )

    items = [
        StoredImportanceResponse(
            feature_name=row.feature_name,
            importance=row.importance,
            method=row.method,
        )
        for row in rows
    ]
    return ImportanceUploadResponse(model_id=model_id, items=items, total=len(items))
