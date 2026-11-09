from uuid import uuid4

import numpy as np
from httpx import AsyncClient

from app.services.drift_engine import compute_psi_from_counts
from app.services.model_service import _compute_histogram


def sample_psi(reference, production, bins=10):
    ref = _compute_histogram(reference, bins=bins)
    prod = _compute_histogram(production, bin_edges=ref["bin_edges"])
    return compute_psi_from_counts(
        *([h["underflow"], *h["counts"], h["overflow"]] for h in (ref, prod))
    )


def _normal(mean: float = 0.0, seed: int = 1, size: int = 1000) -> list[float]:
    return np.random.default_rng(seed).normal(mean, 1.0, size).tolist()


async def _model(client: AsyncClient) -> str:
    response = await client.post("/models", json={"name": f"m-{uuid4()}"})
    assert response.status_code == 201
    return response.json()["id"]


async def _reference(client: AsyncClient, model_id: str, **extra) -> dict:
    body = {
        "model_id": model_id,
        "features": {"a": _normal(seed=1), "b": _normal(seed=2)},
        **extra,
    }
    response = await client.post("/ingest/reference", json=body)
    assert response.status_code == 200, response.text
    return body
