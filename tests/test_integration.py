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
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.models.alert import Alert
from app.models.drift_score import DriftScore
from app.models.feature_importance import FeatureImportance
from app.models.model_registry import ModelRegistry
from app.models.snapshot import Snapshot
from app.services.drift_engine import run_drift_analysis

JANUARY_TS = "2026-01-15T12:00:00Z"
AUGUST_TS = "2026-08-15T12:00:00Z"
AUGUST_WINDOW_DATE = date(2026, 8, 15)
AUGUST_WEEK_START = date(2026, 8, 10)

JAN_REFERENCE_FEATURES = {
    "income": [51000, 52000, 53000, 51500, 52500, 53500, 50500, 52000, 54000, 50000],
    "credit_score": [705, 710, 715, 700, 720, 710, 708, 712, 706, 714],
    "loan_term_months": [36, 36, 36, 36, 36, 36, 36, 36, 36, 36],
}

JAN_SNAPSHOT_FEATURES = {
    "income": [51000, 52000, 53000, 51500, 52500, 53500, 50500, 52000, 54000, 50000],
    "credit_score": [705, 710, 715, 700, 720, 710, 708, 712, 706, 714],
    "loan_term_months": [36, 36, 36, 36, 36, 36, 36, 36, 36, 36],
}

AUG_SNAPSHOT_FEATURES = {
    "income": [36000, 37000, 38000, 39000, 40000, 37500, 36500, 38500, 39500, 35500],
    "credit_score": [635, 640, 645, 650, 655, 648, 642, 638, 652, 646],
    "loan_term_months": [36, 36, 36, 36, 36, 36, 36, 36, 36, 36],
}

FEATURE_IMPORTANCES = {
    "income": 0.45,
    "credit_score": 0.40,
    "loan_term_months": 0.15,
}


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for full integration tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def integration_context() -> AsyncGenerator[tuple[AsyncClient, async_sessionmaker[AsyncSession]], None]:
    """Provide isolated HTTP client and DB session factory for each integration test."""
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


async def _create_model(client: AsyncClient, name: str) -> UUID:
    response = await client.post("/models", json={"name": name, "version": "v3"})
    assert response.status_code == 201
    return UUID(response.json()["id"])


async def _register_reference(client: AsyncClient, model_id: UUID) -> None:
    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": JAN_REFERENCE_FEATURES,
            "feature_importances": FEATURE_IMPORTANCES,
            "importance_method": "manual",
        },
    )
    assert response.status_code == 200


async def _ingest_snapshot(client: AsyncClient, model_id: UUID, timestamp: str, features: dict[str, list[float]]) -> None:
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": timestamp,
            "features": features,
        },
    )
    assert response.status_code == 200


async def _prepare_august_pipeline(client: AsyncClient, model_name: str) -> UUID:
    model_id = await _create_model(client, name=model_name)
    await _register_reference(client, model_id=model_id)
    await _ingest_snapshot(client, model_id=model_id, timestamp=JANUARY_TS, features=JAN_SNAPSHOT_FEATURES)
    await run_drift_analysis_for_date(client, model_id=model_id, target_date=date(2026, 1, 15))
    await _ingest_snapshot(client, model_id=model_id, timestamp=AUGUST_TS, features=AUG_SNAPSHOT_FEATURES)
    return model_id


async def run_drift_analysis_for_date(client: AsyncClient, model_id: UUID, target_date: date) -> None:
    """Run drift analysis synchronously through a direct DB session."""
    override = app.dependency_overrides[get_db]
    async_gen = override()
    db = await anext(async_gen)
    try:
        await run_drift_analysis(db=db, model_id=model_id, window_date=target_date)
    finally:
        await async_gen.aclose()


@pytest.mark.asyncio
async def test_full_pipeline_register_model(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_name = f"loan-scorer-v3-{uuid4()}"

    response = await client.post("/models", json={"name": model_name, "version": "v3"})

    assert response.status_code == 201
    body = response.json()
    model_id = UUID(body["id"])
    assert body["name"] == model_name

    async with session_factory() as session:
        row = await session.execute(select(ModelRegistry).where(ModelRegistry.id == model_id))
        assert row.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_full_pipeline_register_reference(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")

    await _register_reference(client, model_id)

    async with session_factory() as session:
        result = await session.execute(
            select(FeatureImportance).where(FeatureImportance.model_id == model_id)
        )
        rows = result.scalars().all()

    assert len(rows) == 3
    assert sum(row.importance for row in rows) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_full_pipeline_ingest_january_snapshot(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")
    await _register_reference(client, model_id)

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": JANUARY_TS,
            "features": JAN_SNAPSHOT_FEATURES,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["drift_triggered"] is True
    assert set(body["features_recorded"]) == {"income", "credit_score", "loan_term_months"}


@pytest.mark.asyncio
async def test_full_pipeline_drift_january_no_alert(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")
    await _register_reference(client, model_id)
    await _ingest_snapshot(client, model_id, JANUARY_TS, JAN_SNAPSHOT_FEATURES)

    await run_drift_analysis_for_date(client, model_id=model_id, target_date=date(2026, 1, 15))

    async with session_factory() as session:
        result = await session.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == date(2026, 1, 15),
            )
        )
        drift_rows = result.scalars().all()

        alert_rows = await session.execute(select(Alert).where(Alert.model_id == model_id))
        alerts = alert_rows.scalars().all()

    assert len(drift_rows) == 3
    assert all(row.severity == "green" for row in drift_rows)
    assert all(float(row.psi or 0.0) < 0.10 for row in drift_rows)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_full_pipeline_ingest_august_snapshot(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")
    await _register_reference(client, model_id)

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": AUGUST_TS,
            "features": AUG_SNAPSHOT_FEATURES,
        },
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_full_pipeline_drift_august_detects_shift(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")

    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    async with session_factory() as session:
        result = await session.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == AUGUST_WINDOW_DATE,
            )
        )
        rows = {row.feature_name: row for row in result.scalars().all()}

        alerts_result = await session.execute(select(Alert).where(Alert.model_id == model_id))
        alerts = alerts_result.scalars().all()

    assert rows["income"].severity == "red"
    assert float(rows["income"].psi or 0.0) > 0.20
    assert rows["credit_score"].severity in {"red", "yellow"}
    assert rows["loan_term_months"].severity in {"green", "yellow"}
    assert any(alert.alert_type == "drift_red" for alert in alerts)
    assert float(rows["income"].weighted_score or 0.0) > float(rows["loan_term_months"].weighted_score or 0.0)


@pytest.mark.asyncio
async def test_full_pipeline_alerts_created_correctly(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")
    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    response = await client.get(f"/alerts?model_id={model_id}")

    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert any(item["feature_name"] == "income" for item in items)
    assert any("PSI=" in item["message"] for item in items)


@pytest.mark.asyncio
async def test_full_pipeline_resolve_alert(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")
    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    list_response = await client.get(f"/alerts?model_id={model_id}")
    alert_id = list_response.json()["items"][0]["id"]

    resolve_response = await client.patch(f"/alerts/{alert_id}/resolve")
    assert resolve_response.status_code == 200
    assert resolve_response.json()["resolved_at"] is not None

    active_response = await client.get(f"/alerts?model_id={model_id}")
    active_ids = [item["id"] for item in active_response.json()["items"]]
    assert alert_id not in active_ids


@pytest.mark.asyncio
async def test_full_pipeline_generate_report(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")
    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    response = await client.post(
        f"/reports/{model_id}/generate",
        json={"week_start": AUGUST_WEEK_START.isoformat()},
    )

    assert response.status_code == 200
    body = response.json()
    assert float(body["overall_score"] or 0.0) < 80.0
    report_json = body["report_json"]
    assert "executive_summary" in report_json
    assert "feature_drift_summary" in report_json
    assert "output_distribution" in report_json
    assert "top_drifted_features" in report_json
    assert "feature_to_output_correlation" in report_json
    assert "recommendations" in report_json
    assert report_json["top_drifted_features"][0]["feature_name"] == "income"
    assert any(
        "retraining" in recommendation.lower() or "income" in recommendation.lower()
        for recommendation in report_json["recommendations"]
    )


@pytest.mark.asyncio
async def test_full_pipeline_report_markdown_complete(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")
    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)
    await client.post(
        f"/reports/{model_id}/generate",
        json={"week_start": AUGUST_WEEK_START.isoformat()},
    )

    response = await client.get(f"/reports/{model_id}/latest")

    assert response.status_code == 200
    markdown = response.json()["report_markdown"]
    assert isinstance(markdown, str)
    assert markdown.strip()
    assert "## Feature Drift Summary" in markdown
    assert "income" in markdown


@pytest.mark.asyncio
async def test_get_drift_scores_endpoint(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")
    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    response = await client.get(f"/drift/{model_id}/{AUGUST_WINDOW_DATE.isoformat()}")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 3
    assert all(item["severity"] in {"green", "yellow", "red"} for item in items)


@pytest.mark.asyncio
async def test_idempotency_reingest_same_snapshot(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _prepare_august_pipeline(client, model_name=f"loan-scorer-v3-{uuid4()}")

    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    # Reingest the same August snapshot date.
    await _ingest_snapshot(client, model_id=model_id, timestamp=AUGUST_TS, features=AUG_SNAPSHOT_FEATURES)

    async with session_factory() as session:
        snap_result = await session.execute(
            select(Snapshot).where(
                Snapshot.model_id == model_id,
                Snapshot.window_date == AUGUST_WINDOW_DATE,
            )
        )
        snapshot_rows = snap_result.scalars().all()

    assert len(snapshot_rows) == 3

    await run_drift_analysis_for_date(client, model_id=model_id, target_date=AUGUST_WINDOW_DATE)

    async with session_factory() as session:
        drift_result = await session.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == AUGUST_WINDOW_DATE,
            )
        )
        drift_rows = drift_result.scalars().all()

    assert len(drift_rows) == 3
