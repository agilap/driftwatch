import logging

from datetime import date
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal, get_db
from app.routers.dependencies import require_model
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
from app.services.reference_service import list_reference_features, register_reference
from app.services.snapshot_service import (
    get_snapshot_window,
    ingest_snapshot,
    list_snapshot_dates,
)

logger = logging.getLogger(__name__)


async def analyze_snapshot_in_background(model_id: UUID, window_date: date) -> None:
    """Own the session and rollback failed work; persisted snapshots remain retryable."""
    async with AsyncSessionLocal() as db:
        try:
            await run_drift_analysis(db=db, model_id=model_id, window_date=window_date)
        except Exception:
            await db.rollback()
            logger.exception(
                "Drift analysis failed model_id=%s window_date=%s; retry via analyze endpoint",
                model_id,
                window_date,
            )


router = APIRouter(tags=["ingest"])
model_importance_router = APIRouter(tags=["ingest"])


@router.post("/reference", response_model=ReferenceResponse)
async def register_reference_endpoint(
    payload: ReferencePayload,
    db: AsyncSession = Depends(get_db),
) -> ReferenceResponse:
    """Register reference distributions and optional feature importances."""
    return await register_reference(db=db, payload=payload)


@router.get(
    "/reference/{model_id}",
    response_model=ReferenceListResponse,
    dependencies=[Depends(require_model)],
)
async def list_reference_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ReferenceListResponse:
    """List all registered reference features for a model."""
    return await list_reference_features(db=db, model_id=model_id)


@router.post("/snapshot", response_model=SnapshotResponse)
async def ingest_snapshot_endpoint(
    payload: SnapshotPayload,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> SnapshotResponse:
    """Ingest production snapshots and queue drift analysis asynchronously."""
    response = await ingest_snapshot(db=db, payload=payload)
    background_tasks.add_task(
        analyze_snapshot_in_background, response.model_id, response.window_date
    )
    return response


@router.get(
    "/snapshots/{model_id}",
    response_model=list[date],
    dependencies=[Depends(require_model)],
)
async def list_snapshot_dates_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[date]:
    """List all recorded snapshot windows for a model."""
    return await list_snapshot_dates(db=db, model_id=model_id)


@router.get(
    "/snapshots/{model_id}/{window_date}",
    response_model=list[SnapshotFeatureResponse],
    dependencies=[Depends(require_model)],
)
async def get_snapshot_window_endpoint(
    model_id: UUID,
    window_date: date,
    db: AsyncSession = Depends(get_db),
) -> list[SnapshotFeatureResponse]:
    """Get per-feature snapshot stats for a specific model and date window."""
    rows = await get_snapshot_window(db=db, model_id=model_id, window_date=window_date)
    return [SnapshotFeatureResponse.model_validate(row) for row in rows]


@model_importance_router.post(
    "/models/{model_id}/importances",
    response_model=ImportanceUploadResponse,
    dependencies=[Depends(require_model)],
)
async def upload_model_importances_endpoint(
    model_id: UUID,
    payload: ImportanceUploadRequest,
    db: AsyncSession = Depends(get_db),
) -> ImportanceUploadResponse:
    """Upload and upsert model feature importances."""
    normalized = importance_scorer.normalise_importances(payload.importances)

    rows = await importance_scorer.upsert_importances(
        db=db,
        model_id=model_id,
        importances=normalized,
        method=payload.method,
    )

    items = [StoredImportanceResponse.model_validate(row) for row in rows]
    return ImportanceUploadResponse(model_id=model_id, items=items, total=len(items))
