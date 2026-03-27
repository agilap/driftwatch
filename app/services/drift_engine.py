from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from uuid import UUID

import numpy as np
from pydantic import BaseModel
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, ks_2samp
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.drift_score import DriftScore
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot
from app.services import alert_service
from app.services import importance_scorer

logger = logging.getLogger(__name__)

EPSILON = 1e-4
MIN_BIN_SAMPLES_WARNING = 5


class DriftScoreResult(BaseModel):
    """Computed drift metrics for one feature and time window."""

    model_id: UUID
    window_date: date
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    js_divergence: float
    weighted_score: float
    severity: str
    computed_at: datetime


def _validate_inputs(
    reference: np.ndarray, production: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate statistical test inputs and cast to float arrays.

    Args:
        reference: Reference (training) sample array.
        production: Production sample array.

    Returns:
        Tuple of float numpy arrays.

    Raises:
        ValueError: If either input array is empty.
    """
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)

    if ref.size == 0 or prod.size == 0:
        msg = "reference and production arrays must be non-empty"
        raise ValueError(msg)

    return ref, prod


def _build_histogram_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Build shared histogram bin edges using the reference distribution.

    Args:
        reference: Reference (training) sample array.
        bins: Number of bins to generate.

    Returns:
        Array of bin edges with underflow/overflow coverage.
    """
    ref = np.asarray(reference, dtype=float)

    if np.min(ref) == np.max(ref):
        center = float(ref[0])
        spread = 0.5
        edges = np.linspace(center - spread, center + spread, bins + 1)
    else:
        edges = np.histogram_bin_edges(ref, bins=bins)

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _resolve_feature_weights(
    feature_names: list[str],
    importances: dict[str, float],
) -> dict[str, float]:
    """Resolve normalized feature weights from optional importance values.

    Args:
        feature_names: Features to score in current drift window.
        importances: Feature importances keyed by feature name.

    Returns:
        Normalized per-feature weights that sum to 1.0.
    """
    if not feature_names:
        return {}

    if not importances:
        equal = 1.0 / len(feature_names)
        return {feature: equal for feature in feature_names}

    clipped = {
        feature: max(0.0, float(importances.get(feature, 0.0)))
        for feature in feature_names
    }
    total = float(sum(clipped.values()))
    if total <= 0:
        equal = 1.0 / len(feature_names)
        return {feature: equal for feature in feature_names}

    return {feature: value / total for feature, value in clipped.items()}


def _reconstruct_samples_from_histogram(distribution: dict) -> np.ndarray:
    """Approximate sample array from stored histogram bins and counts.

    Args:
        distribution: Histogram payload containing counts and bin_edges.

    Returns:
        Reconstructed sample array using bin midpoints repeated by counts.
    """
    counts = np.asarray(distribution.get("counts", []), dtype=int)
    bin_edges = np.asarray(distribution.get("bin_edges", []), dtype=float)

    if counts.size == 0 or bin_edges.size != counts.size + 1:
        return np.asarray([], dtype=float)

    safe_counts = np.maximum(counts, 0)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return np.repeat(midpoints, safe_counts)


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
    """Compute Population Stability Index between reference and production samples.

    Args:
        reference: Reference (training) sample array.
        production: Production sample array.
        bins: Number of bins. Defaults to 10.

    Returns:
        PSI score clipped to [0, 1].
    """
    ref, prod = _validate_inputs(reference, production)
    edges = _build_histogram_edges(ref, bins=bins)

    ref_counts, _ = np.histogram(ref, bins=edges)
    prod_counts, _ = np.histogram(prod, bins=edges)

    if np.any(ref_counts < MIN_BIN_SAMPLES_WARNING) or np.any(
        prod_counts < MIN_BIN_SAMPLES_WARNING
    ):
        logger.warning(
            "Low histogram bin samples detected for PSI; minimum bin count is %d",
            min(int(np.min(ref_counts)), int(np.min(prod_counts))),
        )

    ref_pct = ref_counts / max(np.sum(ref_counts), 1)
    prod_pct = prod_counts / max(np.sum(prod_counts), 1)

    ref_pct = ref_pct + EPSILON
    prod_pct = prod_pct + EPSILON

    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return float(np.clip(psi, 0.0, 1.0))


def compute_js_divergence(
    reference: np.ndarray, production: np.ndarray, bins: int = 10
) -> float:
    """Compute Jensen-Shannon divergence proxy on binned probability vectors.

    Args:
        reference: Reference (training) sample array.
        production: Production sample array.
        bins: Number of bins. Defaults to 10.

    Returns:
        Jensen-Shannon score clipped to [0, 1].
    """
    ref, prod = _validate_inputs(reference, production)
    edges = _build_histogram_edges(ref, bins=bins)

    ref_counts, _ = np.histogram(ref, bins=edges)
    prod_counts, _ = np.histogram(prod, bins=edges)

    ref_prob = ref_counts / max(np.sum(ref_counts), 1)
    prod_prob = prod_counts / max(np.sum(prod_counts), 1)

    js_score = float(jensenshannon(ref_prob, prod_prob))
    return float(np.clip(js_score, 0.0, 1.0))


def compute_chi_square(
    reference: np.ndarray, production: np.ndarray
) -> dict[str, float | bool]:
    """Compute chi-square goodness-of-fit test for categorical arrays.

    Args:
        reference: Reference categorical sample array.
        production: Production categorical sample array.

    Returns:
        Dictionary with chi-square statistic, p-value, and significance flag.
    """
    ref, prod = _validate_inputs(reference, production)

    ref_categories, ref_counts = np.unique(ref, return_counts=True)
    prod_categories, prod_counts = np.unique(prod, return_counts=True)
    categories = np.union1d(ref_categories, prod_categories)

    ref_map = {
        category: int(count) for category, count in zip(ref_categories, ref_counts)
    }
    prod_map = {
        category: int(count) for category, count in zip(prod_categories, prod_counts)
    }

    aligned_ref = np.asarray(
        [ref_map.get(category, 0) for category in categories], dtype=float
    )
    aligned_prod = np.asarray(
        [prod_map.get(category, 0) for category in categories], dtype=float
    )

    expected = (aligned_ref / max(np.sum(aligned_ref), 1.0)) * np.sum(aligned_prod)
    expected = np.maximum(expected, EPSILON)

    result = chisquare(f_obs=aligned_prod, f_exp=expected)
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

    reference_by_feature = {row.feature_name: row for row in reference_rows}
    snapshot_by_feature = {row.feature_name: row for row in snapshot_rows}
    shared_features = sorted(
        set(reference_by_feature).intersection(snapshot_by_feature)
    )

    raw_importances = await importance_scorer.load_importances(db=db, model_id=model_id)
    if raw_importances:
        feature_weights = importance_scorer.normalise_importances(raw_importances)
    else:
        feature_weights = importance_scorer.equal_weight_importances(shared_features)

    computed_at = datetime.now(timezone.utc)
    results: list[DriftScoreResult] = []
    severity_counts = {"red": 0, "yellow": 0, "green": 0}

    for feature_name in shared_features:
        reference_samples = _reconstruct_samples_from_histogram(
            reference_by_feature[feature_name].distribution
        )
        snapshot_samples = _reconstruct_samples_from_histogram(
            snapshot_by_feature[feature_name].distribution
        )

        if reference_samples.size == 0 or snapshot_samples.size == 0:
            logger.warning(
                "Skipping feature %s due to empty reconstructed samples", feature_name
            )
            continue

        ks = compute_ks_test(reference_samples, snapshot_samples)
        psi = compute_psi(reference_samples, snapshot_samples)
        js_divergence = compute_js_divergence(reference_samples, snapshot_samples)
        severity = classify_severity(psi)
        psi_normalised = float(np.clip(psi, 0.0, 1.0))
        weighted_score = importance_scorer.compute_weighted_score(
            drift_magnitude=psi_normalised,
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
                    ks_statistic=float(ks["statistic"]),
                    ks_pvalue=float(ks["pvalue"]),
                    psi=psi,
                    js_divergence=js_divergence,
                    weighted_score=weighted_score,
                    severity=severity,
                    computed_at=computed_at,
                )
            )
        else:
            existing.ks_statistic = float(ks["statistic"])
            existing.ks_pvalue = float(ks["pvalue"])
            existing.psi = psi
            existing.js_divergence = js_divergence
            existing.weighted_score = weighted_score
            existing.severity = severity
            existing.computed_at = computed_at

        if severity == "red":
            await alert_service.create_alert(
                db=db,
                model_id=model_id,
                feature_name=feature_name,
                alert_type="drift_red",
                severity="red",
                message=(
                    f"Feature '{feature_name}' PSI={psi:.3f} exceeds RED threshold "
                    f"on {window_date}"
                ),
            )

        severity_counts[severity] += 1
        results.append(
            DriftScoreResult(
                model_id=model_id,
                window_date=window_date,
                feature_name=feature_name,
                ks_statistic=float(ks["statistic"]),
                ks_pvalue=float(ks["pvalue"]),
                psi=psi,
                js_divergence=js_divergence,
                weighted_score=weighted_score,
                severity=severity,
                computed_at=computed_at,
            )
        )

    await db.commit()

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
