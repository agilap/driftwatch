from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.feature_importance import FeatureImportance
from app.models.reference_distribution import ReferenceDistribution
from app.schemas.ingest import (
    ReferenceListResponse,
    ReferencePayload,
    ReferenceResponse,
    RegisteredFeatureResponse,
)
from app.services.model_service import _compute_distribution_stats, _compute_histogram

logger = logging.getLogger(__name__)


def _normalise_importances(importances: dict[str, float]) -> dict[str, float]:
    """Normalize feature importances to sum to 1.0."""
    if not importances:
        return {}

    total = float(sum(importances.values()))
    if total <= 0:
        equal_weight = 1.0 / len(importances)
        return {feature: equal_weight for feature in importances}

    return {feature: value / total for feature, value in importances.items()}


async def register_reference(db: AsyncSession, payload: ReferencePayload) -> ReferenceResponse:
    """Register or update reference distributions and optional feature importances."""
    features_registered: list[str] = []

    for feature_name, values in payload.features.items():
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
        normalised = _normalise_importances(payload.feature_importances)

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


async def list_reference_features(db: AsyncSession, model_id: UUID) -> ReferenceListResponse:
    """List registered reference features and stats for a model."""
    result = await db.execute(
        select(ReferenceDistribution)
        .where(ReferenceDistribution.model_id == model_id)
        .order_by(ReferenceDistribution.feature_name.asc())
    )
    rows = result.scalars().all()

    items = [
        RegisteredFeatureResponse(feature_name=row.feature_name, stats=row.stats) for row in rows
    ]

    return ReferenceListResponse(model_id=model_id, items=items, total=len(items))
