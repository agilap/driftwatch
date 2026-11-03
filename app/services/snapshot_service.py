from __future__ import annotations

from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import delete, distinct, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert
from app.models.drift_score import DriftScore
from app.models.health_report import HealthReport
from app.models.snapshot import Snapshot
from app.models.model_registry import ModelRegistry
from app.models.reference_distribution import ReferenceDistribution
from app.schemas.ingest import (
    PREDICTIONS_FEATURE_NAME,
    SnapshotPayload,
    SnapshotResponse,
)
from app.services.model_service import same_summary, summarize_feature


async def ingest_snapshot(
    db: AsyncSession, payload: SnapshotPayload
) -> SnapshotResponse:
    """Ingest daily production feature distributions for a model."""
    model = (
        await db.execute(
            select(ModelRegistry)
            .where(ModelRegistry.id == payload.model_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    references = {
        row.feature_name: row
        for row in (
            await db.execute(
                select(ReferenceDistribution).where(
                    ReferenceDistribution.model_id == payload.model_id
                )
            )
        )
        .scalars()
        .all()
    }
    if not references:
        raise HTTPException(
            status_code=409, detail="Register a reference before snapshots"
        )
    unknown = set(payload.features) - references.keys()
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"Features missing reference: {sorted(unknown)}"
        )

    if payload.predictions is not None and PREDICTIONS_FEATURE_NAME not in references:
        raise HTTPException(
            status_code=422,
            detail="Predictions supplied but the reference has no predictions",
        )

    window_date = payload.timestamp.date()
    features_recorded: list[str] = []
    sample_count = 0

    feature_batches: dict[str, list[float]] = dict(payload.features)
    if payload.predictions is not None:
        feature_batches[PREDICTIONS_FEATURE_NAME] = payload.predictions

    # Validate the whole batch before changing any ORM rows.
    summaries = {}
    for feature_name, values in feature_batches.items():
        if not values:
            raise HTTPException(
                status_code=422, detail=f"Feature '{feature_name}' has no values"
            )
        reference = references.get(feature_name)
        summaries[feature_name] = summarize_feature(
            feature_name,
            values,
            bin_edges=reference.distribution["bin_edges"] if reference else None,
        )

    # A re-post replaces the whole window, so absent features leave it.
    removed = await db.execute(
        delete(Snapshot).where(
            Snapshot.model_id == payload.model_id,
            Snapshot.window_date == window_date,
            Snapshot.feature_name.not_in(list(feature_batches)),
        )
    )
    snapshot_changed = removed.rowcount > 0

    for feature_name, values in feature_batches.items():
        if sample_count == 0:
            sample_count = len(values)

        stats, histogram = summaries[feature_name]

        existing_result = await db.execute(
            select(Snapshot).where(
                Snapshot.model_id == payload.model_id,
                Snapshot.window_date == window_date,
                Snapshot.feature_name == feature_name,
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing is None:
            snapshot_changed = True
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
            snapshot_changed |= not same_summary(
                existing.distribution, histogram
            ) or not same_summary(existing.stats, stats)
            existing.distribution = histogram
            existing.stats = stats
            existing.sample_count = len(values)

        features_recorded.append(feature_name)

    if snapshot_changed:
        # Commit new samples and invalidate their derived evidence atomically.
        # Historical alerts remain audit records: superseded here, they stay
        # queryable as resolved, and analysis raises a new alert if still red.
        now = datetime.now(timezone.utc)
        await db.execute(
            update(Alert)
            .where(
                Alert.model_id == payload.model_id,
                Alert.window_date == window_date,
                Alert.superseded_at.is_(None),
            )
            .values(
                superseded_at=now, resolved_at=func.coalesce(Alert.resolved_at, now)
            )
        )
        await db.execute(
            delete(DriftScore).where(
                DriftScore.model_id == payload.model_id,
                DriftScore.window_date == window_date,
            )
        )
        await db.execute(
            delete(HealthReport).where(
                HealthReport.model_id == payload.model_id,
                HealthReport.week_start <= window_date,
                HealthReport.week_end >= window_date,
            )
        )
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
