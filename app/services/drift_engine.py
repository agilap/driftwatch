from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from uuid import UUID

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.drift_score import DriftScore
from app.models.model_registry import ModelRegistry
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot
from app.schemas.ingest import PREDICTIONS_FEATURE_NAME
from app.services import alert_service
from app.services import importance_scorer
from app.services.model_service import _compute_histogram

logger = logging.getLogger(__name__)

EPSILON = 1e-4


class DriftScoreResult(BaseModel):
    """Computed drift metrics for one feature and time window."""

    model_id: UUID
    window_date: date
    feature_name: str
    ks_statistic: float
    ks_pvalue: float | None = Field(
        description="Null for histogram analysis: binned counts cannot provide a sample KS p-value."
    )
    psi: float
    js_divergence: float
    weighted_score: float
    severity: str
    computed_at: datetime


def _validate_inputs(
    reference: np.ndarray, production: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Require non-empty, finite, one-dimensional numeric sample arrays."""
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)
    for values in (ref, prod):
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("inputs must be non-empty finite one-dimensional arrays")
    return ref, prod


def _validate_counts(
    reference: np.ndarray, production: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Require aligned non-negative counts with positive totals."""
    ref, prod = _validate_inputs(reference, production)
    if ref.size != prod.size or np.any(ref < 0) or np.any(prod < 0):
        raise ValueError("counts must have equal lengths and be non-negative")
    if not np.any(ref > 0) or not np.any(prod > 0):
        raise ValueError("each count vector must have a positive total")
    return ref, prod


def _count_proportions(counts: np.ndarray) -> np.ndarray:
    """Normalize without overflowing the sum of large finite counts."""
    scaled = counts / np.max(counts)
    return scaled / np.sum(scaled)


def compute_psi_from_counts(reference: np.ndarray, production: np.ndarray) -> float:
    """Compute uncapped PSI after EPSILON smoothing and renormalization."""
    ref, prod = _validate_counts(reference, production)
    p = _count_proportions(ref) + EPSILON
    q = _count_proportions(prod) + EPSILON
    p /= np.sum(p)
    q /= np.sum(q)
    return max(0.0, float(np.sum((q - p) * np.log(q / p))))


def compute_js_divergence_from_counts(
    reference: np.ndarray, production: np.ndarray
) -> float:
    """Compute base-2 Jensen-Shannon divergence in [0, 1]."""
    ref, prod = _validate_counts(reference, production)
    p, q = _count_proportions(ref), _count_proportions(prod)
    return float(np.clip(jensenshannon(p, q, base=2) ** 2, 0.0, 1.0))


def compute_binned_ks(reference: np.ndarray, production: np.ndarray) -> float:
    """Return a binned approximation and lower bound on the sample KS statistic.

    Counts must use shared ordered bins, including underflow and overflow bins.
    No sample p-value can be inferred from these histogram counts.
    """
    ref, prod = _validate_counts(reference, production)
    return float(
        np.max(
            np.abs(
                np.cumsum(_count_proportions(ref)) - np.cumsum(_count_proportions(prod))
            )
        )
    )


def _sample_histogram_counts(
    reference: np.ndarray, production: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bin samples on shared reference edges, including out-of-range counts."""
    ref, prod = _validate_inputs(reference, production)
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    reference_histogram = _compute_histogram(ref.tolist(), bins=bins)
    production_histogram = _compute_histogram(
        prod.tolist(), bin_edges=reference_histogram["bin_edges"]
    )
    return tuple(
        np.asarray([hist["underflow"], *hist["counts"], hist["overflow"]], dtype=float)
        for hist in (reference_histogram, production_histogram)
    )


def compute_ks_test(
    reference: np.ndarray, production: np.ndarray
) -> dict[str, float | bool]:
    """Compute two-sample Kolmogorov-Smirnov test.

    Args:
        reference: Reference (training) sample array.
        production: Production sample array.

    Returns:
        Dictionary with KS statistic, p-value, and significance flag.
    """
    ref, prod = _validate_inputs(reference, production)
    result = ks_2samp(ref, prod)
    pvalue = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "pvalue": pvalue,
        "significant": pvalue < settings.ks_pvalue_threshold,
    }


def compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """Compute uncapped PSI from samples using shared reference bins."""
    return compute_psi_from_counts(
        *_sample_histogram_counts(reference, production, bins)
    )


def compute_js_divergence(
    reference: np.ndarray, production: np.ndarray, bins: int = 10
) -> float:
    """Compute base-2 Jensen-Shannon divergence from shared sample histograms."""
    return compute_js_divergence_from_counts(
        *_sample_histogram_counts(reference, production, bins)
    )


def compute_chi_square(
    reference: np.ndarray, production: np.ndarray
) -> dict[str, float | bool]:
    """Compare aligned category count vectors with a contingency-table test."""
    ref, prod = _validate_counts(reference, production)
    table = np.asarray([ref, prod])
    table = table[:, np.any(table > 0, axis=0)]
    if table.shape[1] == 1:
        return {"statistic": 0.0, "pvalue": 1.0, "significant": False}
    result = chi2_contingency(table)
    pvalue = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "pvalue": pvalue,
        "significant": pvalue < settings.ks_pvalue_threshold,
    }


def classify_severity(psi: float) -> str:
    """Classify drift severity using PSI thresholds from settings.

    Args:
        psi: Population Stability Index score.

    Returns:
        Severity level: green, yellow, or red.
    """
    if psi < settings.psi_yellow_threshold:
        return "green"
    if psi <= settings.psi_red_threshold:
        return "yellow"
    return "red"


async def get_drift_scores_for_window(
    db: AsyncSession,
    model_id: UUID,
    window_date: date,
) -> list[DriftScore]:
    """Fetch all drift score rows for one model and date window.

    Args:
        db: Database session.
        model_id: Monitored model identifier.
        window_date: Snapshot date window.

    Returns:
        List of drift score rows ordered by feature name.
    """
    result = await db.execute(
        select(DriftScore)
        .where(DriftScore.model_id == model_id, DriftScore.window_date == window_date)
        .order_by(DriftScore.feature_name.asc())
    )
    return list(result.scalars().all())


async def get_latest_drift_window(db: AsyncSession, model_id: UUID) -> date | None:
    """Fetch the most recently computed drift window for a model.

    Args:
        db: Database session.
        model_id: Monitored model identifier.

    Returns:
        Most recent window date or None.
    """
    result = await db.execute(
        select(DriftScore.window_date)
        .where(DriftScore.model_id == model_id)
        .order_by(DriftScore.window_date.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def run_drift_analysis(
    db: AsyncSession,
    model_id: UUID,
    window_date: date,
) -> list[DriftScoreResult]:
    """Compute and persist drift scores for a model and snapshot window.

    Args:
        db: Database session.
        model_id: Monitored model identifier.
        window_date: Snapshot date window.

    Returns:
        List of computed drift score results for intersecting features.
    """
    # Ingestion uses the same model lock, so analysis sees a complete batch.
    model_result = await db.execute(
        select(ModelRegistry).where(ModelRegistry.id == model_id).with_for_update()
    )
    if model_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Model not found")

    ref_result = await db.execute(
        select(ReferenceDistribution).where(ReferenceDistribution.model_id == model_id)
    )
    reference_rows = list(ref_result.scalars().all())

    snap_result = await db.execute(
        select(Snapshot).where(
            Snapshot.model_id == model_id, Snapshot.window_date == window_date
        )
    )
    snapshot_rows = list(snap_result.scalars().all())

    if not reference_rows:
        raise HTTPException(status_code=409, detail="Reference distribution required")
    if not snapshot_rows:
        raise HTTPException(status_code=404, detail="Snapshot window not found")

    reference_by_feature = {row.feature_name: row for row in reference_rows}
    snapshot_by_feature = {row.feature_name: row for row in snapshot_rows}
    shared_features = sorted(
        set(reference_by_feature).intersection(snapshot_by_feature)
    )

    raw_importances = await importance_scorer.load_importances(db=db, model_id=model_id)
    feature_weights = importance_scorer.resolve_feature_weights(
        [name for name in shared_features if name != PREDICTIONS_FEATURE_NAME],
        raw_importances,
    )

    # Validate every feature before writing scores or alerts for this window.
    for feature_name in shared_features:
        if (
            reference_by_feature[feature_name].distribution["bin_edges"]
            != snapshot_by_feature[feature_name].distribution["bin_edges"]
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Snapshot bin edges differ from reference for {feature_name}; re-ingest this window",
            )

    computed_at = datetime.now(timezone.utc)
    results: list[DriftScoreResult] = []
    severity_counts = {"red": 0, "yellow": 0, "green": 0}
    new_alerts = []

    for feature_name in shared_features:
        reference_histogram = reference_by_feature[feature_name].distribution
        snapshot_histogram = snapshot_by_feature[feature_name].distribution
        reference_counts, snapshot_counts = (
            np.asarray(
                [hist.get("underflow", 0), *hist["counts"], hist.get("overflow", 0)],
                dtype=float,
            )
            for hist in (reference_histogram, snapshot_histogram)
        )
        snapshot_total = (
            sum(snapshot_histogram["counts"])
            + snapshot_histogram.get("underflow", 0)
            + snapshot_histogram.get("overflow", 0)
        )
        if snapshot_total < settings.min_sample_warning:
            logger.warning(
                "Feature '%s' snapshot count %s: PSI is inflated by sampling noise at this size",
                feature_name,
                snapshot_total,
            )
        ks_statistic = compute_binned_ks(reference_counts, snapshot_counts)
        psi = compute_psi_from_counts(reference_counts, snapshot_counts)
        js_divergence = compute_js_divergence_from_counts(
            reference_counts, snapshot_counts
        )
        severity = classify_severity(psi)
        weighted_score = importance_scorer.compute_weighted_score(
            drift_magnitude=min(psi / settings.psi_red_threshold, 1.0),
            feature_importance=feature_weights.get(feature_name, 0.0),
        )

        existing_result = await db.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == window_date,
                DriftScore.feature_name == feature_name,
            )
        )
        existing = existing_result.scalar_one_or_none()

        if existing is None:
            db.add(
                DriftScore(
                    model_id=model_id,
                    window_date=window_date,
                    feature_name=feature_name,
                    ks_statistic=ks_statistic,
                    ks_pvalue=None,
                    psi=psi,
                    js_divergence=js_divergence,
                    weighted_score=weighted_score,
                    severity=severity,
                    computed_at=computed_at,
                )
            )
        else:
            existing.ks_statistic = ks_statistic
            existing.ks_pvalue = None
            existing.psi = psi
            existing.js_divergence = js_divergence
            existing.weighted_score = weighted_score
            existing.severity = severity
            existing.computed_at = computed_at

        if severity == "red":
            label = (
                "Model prediction distribution"
                if feature_name == PREDICTIONS_FEATURE_NAME
                else f"Feature {feature_name!r}"
            )
            alert = await alert_service.create_alert(
                db=db,
                model_id=model_id,
                feature_name=feature_name,
                alert_type=(
                    "output_shift"
                    if feature_name == PREDICTIONS_FEATURE_NAME
                    else "drift_red"
                ),
                severity="red",
                window_date=window_date,
                commit=False,
                message=(
                    f"{label} PSI={psi:.3f} exceeds RED threshold on {window_date}"
                ),
            )

            if getattr(alert, "_dispatch_pending", False):
                new_alerts.append(alert)

        severity_counts[severity] += 1
        results.append(
            DriftScoreResult(
                model_id=model_id,
                window_date=window_date,
                feature_name=feature_name,
                ks_statistic=ks_statistic,
                ks_pvalue=None,
                psi=psi,
                js_divergence=js_divergence,
                weighted_score=weighted_score,
                severity=severity,
                computed_at=computed_at,
            )
        )

    await db.commit()
    for alert in new_alerts:
        await db.refresh(alert)
        await alert_service._dispatch_alert(alert)
        alert._dispatch_pending = False

    logger.info(
        "Drift analysis complete for model=%s window=%s features=%d red=%d yellow=%d green=%d",
        model_id,
        window_date,
        len(results),
        severity_counts["red"],
        severity_counts["yellow"],
        severity_counts["green"],
    )

    return results
