from __future__ import annotations

from numbers import Integral
from uuid import UUID

import numpy as np
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.model_registry import ModelRegistry
from app.schemas.model import ModelListResponse, ModelResponse


async def create_model(
    db: AsyncSession, name: str, version: str | None
) -> ModelResponse:
    """Create and persist a new model registry record."""
    model = ModelRegistry(name=name, version=version)
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return ModelResponse.model_validate(model)


async def get_model(db: AsyncSession, model_id: UUID) -> ModelResponse | None:
    """Fetch a model by ID."""
    result = await db.execute(select(ModelRegistry).where(ModelRegistry.id == model_id))
    model = result.scalar_one_or_none()
    if model is None:
        return None
    return ModelResponse.model_validate(model)


async def list_models(db: AsyncSession, page: int, page_size: int) -> ModelListResponse:
    """Return paginated models and total count."""
    offset = (page - 1) * page_size

    total_result = await db.execute(select(func.count()).select_from(ModelRegistry))
    total = int(total_result.scalar_one())

    items_result = await db.execute(
        select(ModelRegistry)
        .order_by(ModelRegistry.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = [
        ModelResponse.model_validate(item) for item in items_result.scalars().all()
    ]

    return ModelListResponse(items=items, total=total)


def _compute_distribution_stats(values: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for a numeric feature array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Values must be a non-empty, finite one-dimensional array")

    with np.errstate(over="ignore", invalid="ignore"):
        stats = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }
    if not all(np.isfinite(value) for value in stats.values()):
        raise ValueError("values overflow when computing statistics")
    return stats


def _compute_histogram(
    values: list[float], bins: int = 10, bin_edges: list[float] | None = None
) -> dict[str, list[float] | list[int] | int]:
    """Count numeric observations on finite edges, retaining out-of-range values."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Values must be a non-empty, finite one-dimensional array")
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be a positive integer")

    edges = (
        np.histogram_bin_edges(arr, bins=bins)
        if bin_edges is None
        else np.asarray(bin_edges, dtype=float)
    )
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("bin_edges must be finite and strictly increasing")
    counts, _ = np.histogram(arr, bins=edges)

    return {
        "bins": list(range(len(counts))),
        "counts": counts.astype(int).tolist(),
        "bin_edges": edges.astype(float).tolist(),
        "underflow": int(np.count_nonzero(arr < edges[0])),
        "overflow": int(np.count_nonzero(arr > edges[-1])),
    }


def _float_leaves(value):
    if isinstance(value, dict):
        return {key: _float_leaves(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_float_leaves(item) for item in value]
    return float(value)


def same_summary(stored: dict, computed: dict) -> bool:
    """Compare stats or histograms; JSONB returns large floats like 3e+23 as ints."""
    return _float_leaves(stored) == _float_leaves(computed)


def summarize_feature(
    feature_name: str, values: list[float], bin_edges: list[float] | None = None
) -> tuple[dict[str, float], dict[str, list[float] | list[int] | int]]:
    """Return stats and histogram, or a 422 for samples they cannot represent."""
    try:
        return (
            _compute_distribution_stats(values),
            _compute_histogram(values, bin_edges=bin_edges),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Feature '{feature_name}' has values too extreme to summarize: {exc}",
        ) from exc
