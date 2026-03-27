from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from unittest.mock import patch
from uuid import UUID

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.models.feature_importance import FeatureImportance
from app.models.reference_distribution import ReferenceDistribution


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for ingest endpoint tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def ingest_context() -> (
    AsyncGenerator[tuple[AsyncClient, async_sessionmaker[AsyncSession]], None]
):
    """Provide HTTP client and session factory bound to a real test DB."""
    engine = create_async_engine(_database_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        await session.execute(
            text(
                "TRUNCATE TABLE "
                "alerts, health_reports, drift_scores, snapshots, "
                "feature_importances, reference_distributions, models "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory

    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


async def _create_model(
    client: AsyncClient, name: str = "loan-scorer", version: str = "v1"
) -> UUID:
    response = await client.post("/models", json={"name": name, "version": version})
    assert response.status_code == 201
    return UUID(response.json()["id"])


@pytest.mark.asyncio
async def test_register_reference_success(
    ingest_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_context
    model_id = await _create_model(client, name="model-success")

    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {
                "income": [1, 2, 3, 4, 5],
                "credit_score": [600, 650, 700, 750, 800],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == str(model_id)
    assert set(body["features_registered"]) == {"income", "credit_score"}
    assert body["importances_registered"] is False

    async with session_factory() as session:
        result = await session.execute(
            select(ReferenceDistribution).where(
                ReferenceDistribution.model_id == model_id
            )
        )
        rows = result.scalars().all()

    assert len(rows) == 2
    assert all("mean" in row.stats for row in rows)
    assert all("counts" in row.distribution for row in rows)


@pytest.mark.asyncio
async def test_register_reference_unknown_model(
    ingest_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = ingest_context

    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": "00000000-0000-0000-0000-000000000001",
            "features": {"income": [1, 2, 3]},
        },
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_register_reference_with_importances(
    ingest_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_context
    model_id = await _create_model(client, name="model-importances")

    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {
                "income": [10, 20, 30, 40],
                "credit_score": [600, 650, 700, 720],
            },
            "feature_importances": {"income": 2.0, "credit_score": 1.0},
            "importance_method": "manual",
        },
    )

    assert response.status_code == 200
    assert response.json()["importances_registered"] is True

    async with session_factory() as session:
        result = await session.execute(
            select(FeatureImportance).where(FeatureImportance.model_id == model_id)
        )
        importances = result.scalars().all()

    assert len(importances) == 2
    assert sum(item.importance for item in importances) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_register_reference_low_sample_count(
    ingest_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = ingest_context
    model_id = await _create_model(client, name="model-low-sample")

    with patch("app.services.reference_service.logger.warning") as warning_mock:
        response = await client.post(
            "/ingest/reference",
            json={
                "model_id": str(model_id),
                "features": {"income": [1.0, 2.0, 3.0]},
            },
        )

    assert response.status_code == 200
    warning_mock.assert_called_once()


@pytest.mark.asyncio
async def test_register_reference_upsert(
    ingest_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_context
    model_id = await _create_model(client, name="model-upsert")

    first = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {"income": [1, 1, 1, 1, 1]},
        },
    )
    assert first.status_code == 200

    second = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {"income": [10, 20, 30, 40, 50]},
        },
    )
    assert second.status_code == 200

    async with session_factory() as session:
        result = await session.execute(
            select(ReferenceDistribution).where(
                ReferenceDistribution.model_id == model_id,
                ReferenceDistribution.feature_name == "income",
            )
        )
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].stats["mean"] == pytest.approx(30.0)
