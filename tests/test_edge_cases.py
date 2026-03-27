from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import date
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.services.drift_engine import compute_psi


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for edge-case integration tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def edge_context() -> AsyncGenerator[AsyncClient, None]:
    """Provide isolated HTTP client with test database dependency override."""
    engine = create_async_engine(_database_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_empty_features_dict_rejected(edge_context: AsyncClient) -> None:
    client = edge_context
    model_response = await client.post("/models", json={"name": f"edge-model-{uuid4()}", "version": "v1"})
    model_id = model_response.json()["id"]

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": model_id,
            "timestamp": "2026-03-27T00:00:00Z",
            "features": {"income": []},
        },
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_model_not_found_all_endpoints(edge_context: AsyncClient) -> None:
    client = edge_context
    missing_model_id = "00000000-0000-0000-0000-000000000001"

    checks = [
        ("get", f"/models/{missing_model_id}", None),
        ("post", "/ingest/reference", {
            "model_id": missing_model_id,
            "features": {"income": [1.0, 2.0, 3.0]},
        }),
        ("post", "/ingest/snapshot", {
            "model_id": missing_model_id,
            "timestamp": "2026-03-27T00:00:00Z",
            "features": {"income": [1.0, 2.0, 3.0]},
        }),
        ("get", f"/ingest/snapshots/{missing_model_id}", None),
        ("get", f"/ingest/snapshots/{missing_model_id}/2026-03-27", None),
        ("get", f"/drift/{missing_model_id}/2026-03-27", None),
        ("post", f"/models/{missing_model_id}/importances", {
            "importances": {"income": 1.0},
            "method": "manual",
        }),
        ("get", f"/reports/{missing_model_id}", None),
        ("get", f"/reports/{missing_model_id}/latest", None),
        ("get", f"/reports/{missing_model_id}/2026-03-24", None),
        ("post", f"/reports/{missing_model_id}/generate", {"week_start": "2026-03-24"}),
    ]

    for method, path, payload in checks:
        if method == "get":
            response = await client.get(path)
        else:
            response = await client.post(path, json=payload)
        assert response.status_code == 404


def test_single_value_feature_psi() -> None:
    psi = compute_psi(reference=[1.0], production=[1.0], bins=10)  # type: ignore[arg-type]
    assert psi >= 0.0


def test_identical_distributions_psi_near_zero() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    psi = compute_psi(reference=values, production=values, bins=10)  # type: ignore[arg-type]
    assert psi < 1e-6


def test_completely_different_distributions_psi_high() -> None:
    reference = [1.0] * 200
    production = [100.0] * 200
    psi = compute_psi(reference=reference, production=production, bins=10)  # type: ignore[arg-type]
    assert psi > 0.20
