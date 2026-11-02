from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.feature_importance import FeatureImportance
from app.models.model_registry import ModelRegistry
from app.models.snapshot import Snapshot
from app.models.reference_distribution import ReferenceDistribution
from app.schemas.ingest import (
    PREDICTIONS_FEATURE_NAME,
    ReferenceListResponse,
    ReferencePayload,
    ReferenceResponse,
    RegisteredFeatureResponse,
)
from app.services import importance_scorer
from app.services.model_service import _compute_distribution_stats, _compute_histogram

logger = logging.getLogger(__name__)


async def register_reference(
    db: AsyncSession, payload: ReferencePayload
) -> ReferenceResponse:
    """Register or update reference distributions and optional feature importances."""
    model = (
        await db.execute(
            select(ModelRegistry)
            .where(ModelRegistry.id == payload.model_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    frozen = (
        await db.execute(
            select(Snapshot.id).where(Snapshot.model_id == payload.model_id).limit(1)
        )
    ).first() is not None
    feature_batches = dict(payload.features)
    if payload.predictions is not None:
        feature_batches[PREDICTIONS_FEATURE_NAME] = payload.predictions

    # Validate the whole request before changing any ORM rows.
    if frozen:
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
        for name, values in feature_batches.items():
            row = references.get(name)
            if (
                row is None
                or row.distribution != _compute_histogram(values)
                or row.stats != _compute_distribution_stats(values)
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Reference is immutable after snapshots; register a new model version",
                )
    features_registered: list[str] = []

    for feature_name, values in feature_batches.items():
        if len(values) < settings.min_sample_warning:
            logger.warning(
                "Feature '%s' has only %d samples; below warning threshold %d",
                feature_name,
                len(values),
                settings.min_sample_warning,
            )

        stats = _compute_distribution_stats(values)
        histogram = _compute_histogram(values)

        existing_result = await db.execute(
            select(ReferenceDistribution).where(
                ReferenceDistribution.model_id == payload.model_id,
                ReferenceDistribution.feature_name == feature_name,
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing is None:
            db.add(
                ReferenceDistribution(
                    model_id=payload.model_id,
                    feature_name=feature_name,
                    distribution=histogram,
                    stats=stats,
                )
            )
        else:
            existing.distribution = histogram
            existing.stats = stats

        features_registered.append(feature_name)

    importances_registered = bool(payload.feature_importances)
    if payload.feature_importances:
        normalised = importance_scorer.import_from_json(
            json_data=payload.feature_importances,
            method=payload.importance_method or "manual",
        )

        for feature_name, importance in normalised.items():
            existing_result = await db.execute(
                select(FeatureImportance).where(
                    FeatureImportance.model_id == payload.model_id,
                    FeatureImportance.feature_name == feature_name,
                )
            )
            existing = existing_result.scalar_one_or_none()

            if existing is None:
                db.add(
                    FeatureImportance(
                        model_id=payload.model_id,
                        feature_name=feature_name,
                        importance=importance,
                        method=payload.importance_method,
                    )
                )
            else:
                existing.importance = importance
                existing.method = payload.importance_method

    await db.commit()

    return ReferenceResponse(
        model_id=payload.model_id,
        features_registered=sorted(features_registered),
        importances_registered=importances_registered,
        registered_at=datetime.now(timezone.utc),
    )


async def list_reference_features(
    db: AsyncSession, model_id: UUID
) -> ReferenceListResponse:
    """List registered reference features and stats for a model."""
    result = await db.execute(
        select(ReferenceDistribution)
        .where(ReferenceDistribution.model_id == model_id)
        .order_by(ReferenceDistribution.feature_name.asc())
    )
    rows = result.scalars().all()

    items = [
        RegisteredFeatureResponse(feature_name=row.feature_name, stats=row.stats)
        for row in rows
    ]

    return ReferenceListResponse(model_id=model_id, items=items, total=len(items))
