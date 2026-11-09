from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from datetime import date
from uuid import UUID

import pytest

from tests.conftest import truncate_tables
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot


def _database_url() -> str:
    return os.environ["DATABASE_URL"]


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
        await truncate_tables(session)
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
    model_id = UUID(response.json()["id"])
    reference = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {
                "income": [10.0, 20.0, 30.0],
                "credit_score": [600.0, 620.0, 640.0],
            },
        },
    )
    assert reference.status_code == 200
    return model_id


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
    reference = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {"income": [10.0, 20.0, 30.0]},
            "predictions": [0.1, 0.2, 0.3],
        },
    )
    assert reference.status_code == 200

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
async def test_ingest_snapshot_responds_before_analysis(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-speed")

    events = []
    analysis_calls = []

    async def record_analysis(model_id: UUID, window_date: date) -> None:
        events.append("analysis")
        analysis_calls.append((model_id, window_date))

    monkeypatch.setattr(
        "app.routers.ingest.analyze_snapshot_in_background", record_analysis
    )
    body = json.dumps(
        {
            "model_id": str(model_id),
            "timestamp": "2026-03-26T12:00:00Z",
            "features": {"income": [10.0, 20.0, 30.0]},
        }
    ).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ingest/snapshot",
        "raw_path": b"/ingest/snapshot",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    request_sent = False
    response_complete = asyncio.Event()
    messages = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await response_complete.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        events.append(message["type"])
        messages.append(message)
        if message["type"] == "http.response.body" and not message.get(
            "more_body", False
        ):
            response_complete.set()

    await app(scope, receive, send)

    response_start = next(m for m in messages if m["type"] == "http.response.start")
    assert response_start["status"] == 200
    assert analysis_calls == [(model_id, date(2026, 3, 26))]
    assert events.index("http.response.start") < events.index("analysis")


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


@pytest.mark.asyncio
async def test_snapshot_uses_reference_edges_and_preserves_outliers_on_retry(
    ingest_snapshot_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = ingest_snapshot_context
    model_id = await _create_model(client, name="snapshot-shared-edges")
    payload = {
        "model_id": str(model_id),
        "timestamp": "2026-03-26T12:00:00Z",
        "features": {"income": [-10.0, 10.0, 20.0, 30.0, 50.0]},
    }
    first = await client.post("/ingest/snapshot", json=payload)
    retry = await client.post("/ingest/snapshot", json=payload)
    assert first.status_code == retry.status_code == 200

    async with session_factory() as session:
        reference = (
            await session.execute(
                select(ReferenceDistribution).where(
                    ReferenceDistribution.model_id == model_id,
                    ReferenceDistribution.feature_name == "income",
                )
            )
        ).scalar_one()
        snapshots = (
            (
                await session.execute(
                    select(Snapshot).where(
                        Snapshot.model_id == model_id,
                        Snapshot.feature_name == "income",
                    )
                )
            )
            .scalars()
            .all()
        )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.distribution["bin_edges"] == reference.distribution["bin_edges"]
    assert reference.distribution["underflow"] == 0
    assert reference.distribution["overflow"] == 0
    assert sum(reference.distribution["counts"]) == 3
    assert snapshot.distribution["underflow"] == 1
    assert snapshot.distribution["overflow"] == 1
    assert sum(snapshot.distribution["counts"]) == 3
    assert (
        (
            sum(snapshot.distribution["counts"])
            + snapshot.distribution["underflow"]
            + snapshot.distribution["overflow"]
        )
        == snapshot.sample_count
        == 5
    )

    identical_reference = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {"income": [10.0, 20.0, 30.0]},
        },
    )
    changed_reference = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": {"income": [10.0, 20.0, 31.0]},
        },
    )
    assert identical_reference.status_code == 200
    assert changed_reference.status_code == 409
