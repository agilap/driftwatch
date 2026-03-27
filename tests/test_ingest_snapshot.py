from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator
from datetime import date
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
from app.models.snapshot import Snapshot


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for snapshot ingest endpoint tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def ingest_snapshot_context() -> (
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
async def test_ingest_snapshot_success(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-success")

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-26T23:59:00Z",
            "features": {
                "income": [10.0, 20.0, 30.0],
                "credit_score": [600.0, 620.0, 640.0],
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == str(model_id)
    assert body["window_date"] == "2026-03-26"
    assert set(body["features_recorded"]) == {"income", "credit_score"}
    assert body["sample_count"] == 3
    assert body["drift_triggered"] is True

    async with session_factory() as session:
        result = await session.execute(
            select(Snapshot).where(
                Snapshot.model_id == model_id,
                Snapshot.window_date == date(2026, 3, 26),
            )
        )
        rows = result.scalars().all()

    assert len(rows) == 2


@pytest.mark.asyncio
async def test_ingest_snapshot_unknown_model(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = ingest_snapshot_context

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": "00000000-0000-0000-0000-000000000001",
            "timestamp": "2026-03-26T12:00:00Z",
            "features": {"income": [10.0, 20.0, 30.0]},
        },
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_ingest_snapshot_predictions_stored(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-preds")

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-26T12:00:00Z",
            "features": {"income": [10.0, 20.0, 30.0]},
            "predictions": [0.1, 0.2, 0.3],
        },
    )

    assert response.status_code == 200
    assert "__predictions__" in response.json()["features_recorded"]

    async with session_factory() as session:
        result = await session.execute(
            select(Snapshot).where(
                Snapshot.model_id == model_id,
                Snapshot.feature_name == "__predictions__",
            )
        )
        row = result.scalar_one_or_none()

    assert row is not None
    assert row.sample_count == 3


@pytest.mark.asyncio
async def test_ingest_snapshot_upsert(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-upsert")

    first = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-26T00:00:00Z",
            "features": {"income": [1.0, 1.0, 1.0]},
        },
    )
    assert first.status_code == 200

    second = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-26T23:59:59Z",
            "features": {"income": [10.0, 20.0, 30.0]},
        },
    )
    assert second.status_code == 200

    async with session_factory() as session:
        result = await session.execute(
            select(Snapshot).where(
                Snapshot.model_id == model_id,
                Snapshot.feature_name == "income",
            )
        )
        rows = result.scalars().all()

    assert len(rows) == 1
    assert rows[0].stats["mean"] == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_ingest_snapshot_returns_immediately(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-speed")

    start = time.perf_counter()
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-26T12:00:00Z",
            "features": {"income": [10.0, 20.0, 30.0]},
        },
    )
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    assert elapsed < 0.2


@pytest.mark.asyncio
async def test_list_snapshot_dates(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-dates")

    first = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-25T23:59:59Z",
            "features": {"income": [1.0, 2.0, 3.0]},
        },
    )
    assert first.status_code == 200

    second = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": "2026-03-26T00:00:01Z",
            "features": {"income": [3.0, 4.0, 5.0]},
        },
    )
    assert second.status_code == 200

    response = await client.get(f"/ingest/snapshots/{model_id}")

    assert response.status_code == 200
    assert response.json() == ["2026-03-26", "2026-03-25"]
