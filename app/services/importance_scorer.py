from __future__ import annotations

from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.feature_importance import FeatureImportance
from app.models.model_registry import ModelRegistry

ALLOWED_METHODS = {"shap", "coefficient", "manual"}


async def load_importances(db: AsyncSession, model_id: UUID) -> dict[str, float]:
    """Load stored feature importances for a model.

    Args:
            db: Database session.
            model_id: Monitored model identifier.

    Returns:
            Mapping of feature name to raw importance value.
    """
    result = await db.execute(
        select(FeatureImportance).where(FeatureImportance.model_id == model_id)
    )
    rows = list(result.scalars().all())
    if not rows:
        return {}
    return {row.feature_name: float(row.importance) for row in rows}


def normalise_importances(importances: dict[str, float]) -> dict[str, float]:
    """Normalise feature importances so values sum to exactly 1.0.

    Args:
            importances: Raw feature importances.

    Returns:
            Normalized feature importances.

    Raises:
            ValueError: If any value is negative or all values are zero.
    """
    if not importances:
        return {}

    if any(not np.isfinite(value) or value < 0 for value in importances.values()):
        msg = "importance values must be finite and non-negative"
        raise ValueError(msg)

    total = float(sum(importances.values()))
    if not np.isfinite(total) or total <= 0:
        msg = "at least one importance value must be greater than zero"
        raise ValueError(msg)

    normalized = {
        feature: float(value) / total for feature, value in importances.items()
    }
    # Adjust for floating-point residue to keep an exact 1.0 sum.
    residue = 1.0 - sum(normalized.values())
    if normalized:
        largest_key = max(normalized, key=normalized.get)
        normalized[largest_key] = normalized[largest_key] + residue
    return normalized


def equal_weight_importances(feature_names: list[str]) -> dict[str, float]:
    """Generate equal feature weights.

    Args:
            feature_names: Feature names to weight.

    Returns:
            Mapping with each feature assigned 1/N.
    """
    if not feature_names:
        return {}
    weight = 1.0 / len(feature_names)
    return {feature_name: weight for feature_name in feature_names}


def resolve_feature_weights(
    feature_names: list[str], importances: dict[str, float]
) -> dict[str, float]:
    """Normalize importance over analyzed features, with equal-weight fallback."""
    selected = {feature: importances.get(feature, 0.0) for feature in feature_names}
    if any(not np.isfinite(value) or value < 0 for value in selected.values()):
        raise ValueError("importance values must be finite and non-negative")
    if not any(value > 0 for value in selected.values()):
        return equal_weight_importances(feature_names)
    return normalise_importances(selected)


def compute_weighted_score(drift_magnitude: float, feature_importance: float) -> float:
    """Compute weighted drift score from drift magnitude and feature importance.

    Args:
            drift_magnitude: Normalized drift magnitude in [0, 1].
            feature_importance: Normalized feature importance in [0, 1].

    Returns:
            Weighted score in [0, 1].
    """
    if any(
        not np.isfinite(value) or value < 0
        for value in (drift_magnitude, feature_importance)
    ):
        raise ValueError(
            "drift magnitude and importance must be finite and non-negative"
        )
    weighted = float(drift_magnitude) * float(feature_importance)
    return float(np.clip(weighted, 0.0, 1.0))


def rank_features_by_weighted_score(
    drift_scores: list[dict],
    importances: dict[str, float],
) -> list[dict]:
    """Rank feature drift rows by weighted score.

    Args:
            drift_scores: List of feature drift dictionaries containing at least
                    feature_name and psi fields.
            importances: Feature importance map.

    Returns:
            Drift rows enriched with weighted_score and rank, sorted descending.
    """
    if not drift_scores:
        return []

    feature_names = [str(row["feature_name"]) for row in drift_scores]
    weights = resolve_feature_weights(feature_names, importances)

    ranked: list[dict] = []
    for row in drift_scores:
        feature_name = str(row["feature_name"])
        psi = max(0.0, float(row.get("psi", 0.0)))
        psi_normalised = min(psi / settings.psi_red_threshold, 1.0)
        weighted_score = compute_weighted_score(
            psi_normalised, weights.get(feature_name, 0.0)
        )

        enriched = dict(row)
        enriched["weighted_score"] = weighted_score
        ranked.append(enriched)

    ranked.sort(key=lambda item: float(item["weighted_score"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index

    return ranked


def import_from_json(json_data: dict, method: str) -> dict[str, float]:
    """Validate and normalize feature importances from JSON input.

    Args:
            json_data: Raw mapping of feature_name to importance value.
            method: Importance extraction method.

    Returns:
            Normalized importances.

    Raises:
            ValueError: If method is invalid or importances are malformed.
    """
    if method not in ALLOWED_METHODS:
        msg = "method must be one of: shap, coefficient, manual"
        raise ValueError(msg)

    if not json_data:
        msg = "importances payload cannot be empty"
        raise ValueError(msg)

    parsed: dict[str, float] = {}
    for feature_name, value in json_data.items():
        if not isinstance(value, (int, float)):
            msg = f"importance for feature '{feature_name}' must be numeric"
            raise ValueError(msg)
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0:
            msg = f"importance for feature '{feature_name}' must be non-negative"
            raise ValueError(msg)
        parsed[str(feature_name)] = numeric

    return normalise_importances(parsed)


async def upsert_importances(
    db: AsyncSession,
    model_id: UUID,
    importances: dict[str, float],
    method: str,
) -> list[FeatureImportance]:
    """Upsert normalized feature importances for a model.

    Args:
            db: Database session.
            model_id: Monitored model identifier.
            importances: Normalized feature importances.
            method: Method used to derive importances.

    Returns:
            List of persisted feature importance rows.
    """
    await db.execute(
        select(ModelRegistry.id).where(ModelRegistry.id == model_id).with_for_update()
    )
    persisted: list[FeatureImportance] = []
    for feature_name, importance in importances.items():
        existing_result = await db.execute(
            select(FeatureImportance).where(
                FeatureImportance.model_id == model_id,
                FeatureImportance.feature_name == feature_name,
            )
        )
        existing = existing_result.scalar_one_or_none()
        if existing is None:
            row = FeatureImportance(
                model_id=model_id,
                feature_name=feature_name,
                importance=importance,
                method=method,
            )
            db.add(row)
            persisted.append(row)
        else:
            existing.importance = importance
            existing.method = method
            persisted.append(existing)

    await db.commit()

    for row in persisted:
        await db.refresh(row)

    persisted.sort(key=lambda row: row.feature_name)
    return persisted
