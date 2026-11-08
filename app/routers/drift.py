from __future__ import annotations

from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routers.dependencies import require_model
from app.schemas.drift import DriftScoreResponse, FeatureDriftSummary, ModelDriftSummary
from app.schemas.ingest import PREDICTIONS_FEATURE_NAME
from app.services.drift_engine import (
    DriftScoreResult,
    get_drift_scores_for_window,
    get_latest_drift_window,
    run_drift_analysis,
)

router = APIRouter(tags=["drift"])


@router.get(
    "/{model_id}/latest",
    response_model=ModelDriftSummary,
    dependencies=[Depends(require_model)],
)
async def get_latest_drift_window_endpoint(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ModelDriftSummary:
    """Return summary for the most recently computed drift window for a model."""
    latest_window = await get_latest_drift_window(db=db, model_id=model_id)
    if latest_window is None:
        raise HTTPException(status_code=404, detail="No drift scores found for model")

    rows = await get_drift_scores_for_window(
        db=db, model_id=model_id, window_date=latest_window
    )

    features: list[FeatureDriftSummary] = []
    red_count = 0
    yellow_count = 0
    green_count = 0

    for row in rows:
        psi = float(row.psi or 0.0)
        weighted_score = float(row.weighted_score or 0.0)
        severity = row.severity or "green"

        features.append(
            FeatureDriftSummary(
                feature_name=row.feature_name,
                psi=psi,
                severity=severity,
                weighted_score=weighted_score,
            )
        )

        if row.feature_name == PREDICTIONS_FEATURE_NAME:
            continue

        if severity == "red":
            red_count += 1
        elif severity == "yellow":
            yellow_count += 1
        else:
            green_count += 1

    return ModelDriftSummary(
        model_id=model_id,
        window_date=latest_window,
        features=features,
        red_count=red_count,
        yellow_count=yellow_count,
        green_count=green_count,
    )


@router.get(
    "/{model_id}/{window_date}",
    response_model=list[DriftScoreResponse],
    dependencies=[Depends(require_model)],
)
async def get_drift_window_endpoint(
    model_id: UUID,
    window_date: date,
    db: AsyncSession = Depends(get_db),
) -> list[DriftScoreResponse]:
    """Return drift score rows for a model and specific date window."""
    rows = await get_drift_scores_for_window(
        db=db, model_id=model_id, window_date=window_date
    )
    return [DriftScoreResponse.model_validate(row) for row in rows]


@router.post("/{model_id}/{window_date}/analyze", response_model=list[DriftScoreResult])
async def retry_drift_analysis_endpoint(
    model_id: UUID,
    window_date: date,
    db: AsyncSession = Depends(get_db),
) -> list[DriftScoreResult]:
    """Recompute a committed snapshot window, including after background failure."""
    return await run_drift_analysis(db=db, model_id=model_id, window_date=window_date)
