from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.snapshot import Snapshot
from app.schemas.ingest import SnapshotPayload, SnapshotResponse
from app.services.model_service import (
    _compute_distribution_stats,
    _compute_histogram,
    get_model,
)

PREDICTIONS_FEATURE_NAME = "__predictions__"


async def ingest_snapshot(
    db: AsyncSession, payload: SnapshotPayload
) -> SnapshotResponse:
    """Ingest daily production feature distributions for a model."""
    model = await get_model(db=db, model_id=payload.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    window_date = payload.timestamp.date()
    features_recorded: list[str] = []
    sample_count = 0

    feature_batches: dict[str, list[float]] = dict(payload.features)
    if payload.predictions is not None:
        feature_batches[PREDICTIONS_FEATURE_NAME] = payload.predictions

    for feature_name, values in feature_batches.items():
        if not values:
            raise HTTPException(
                status_code=422, detail=f"Feature '{feature_name}' has no values"
            )

        if sample_count == 0:
            sample_count = len(values)

        stats = _compute_distribution_stats(values)
        histogram = _compute_histogram(values)

        existing_result = await db.execute(
            select(Snapshot).where(
                Snapshot.model_id == payload.model_id,
                Snapshot.window_date == window_date,
                Snapshot.feature_name == feature_name,
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing is None:
            db.add(
                Snapshot(
                    model_id=payload.model_id,
                    window_date=window_date,
                    feature_name=feature_name,
                    distribution=histogram,
                    stats=stats,
                    sample_count=len(values),
                )
            )
        else:
            existing.distribution = histogram
            existing.stats = stats
            existing.sample_count = len(values)

        features_recorded.append(feature_name)

    await db.commit()

    return SnapshotResponse(
        model_id=payload.model_id,
        window_date=window_date,
        features_recorded=sorted(features_recorded),
        sample_count=sample_count,
        drift_triggered=True,
        recorded_at=datetime.now(timezone.utc),
    )


async def get_snapshot_window(
    db: AsyncSession,
    model_id: UUID,
    window_date: date,
) -> list[Snapshot]:
    """Return snapshot rows for a specific model and date window."""
    rows_result = await db.execute(
        select(Snapshot)
        .where(Snapshot.model_id == model_id, Snapshot.window_date == window_date)
        .order_by(Snapshot.feature_name.asc())
    )
    return list(rows_result.scalars().all())


async def list_snapshot_dates(db: AsyncSession, model_id: UUID) -> list[date]:
    """List all recorded snapshot window dates for a model."""
    rows_result = await db.execute(
        select(distinct(Snapshot.window_date))
        .where(Snapshot.model_id == model_id)
        .order_by(Snapshot.window_date.desc())
    )
    return [row[0] for row in rows_result.all()]
