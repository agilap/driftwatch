from __future__ import annotations

import os
from datetime import date
from collections.abc import AsyncGenerator
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.alert import Alert
from app.models.model_registry import ModelRegistry
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot
from app.services.alert_service import (
    _dispatch_alert,
    create_alert,
    list_alerts,
    resolve_alert,
)
from app.services.drift_engine import run_drift_analysis
from app.services.model_service import _compute_distribution_stats, _compute_histogram


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for alert service tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide isolated async DB session for each alert test."""
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
        yield session

    await engine.dispose()


async def _create_model(db_session: AsyncSession, name: str | None = None) -> UUID:
    model = ModelRegistry(name=name or f"model-{uuid4()}", version="v1")
    db_session.add(model)
    await db_session.commit()
    await db_session.refresh(model)
    return model.id


@pytest.mark.asyncio
async def test_create_alert_stored_in_db(db_session: AsyncSession) -> None:
    model_id = await _create_model(db_session)

    alert = await create_alert(
        db=db_session,
        model_id=model_id,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift detected",
    )

    persisted = await db_session.execute(select(Alert).where(Alert.id == alert.id))
    row = persisted.scalar_one_or_none()

    assert row is not None
    assert row.alert_type == "drift_red"
    assert row.severity == "red"


@pytest.mark.asyncio
async def test_create_alert_red_drift_triggers_on_red_severity(
    db_session: AsyncSession,
) -> None:
    model_id = await _create_model(db_session)

    reference_values = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    production_values = [12.0, 13.0, 14.0, 15.0, 16.0] * 20

    db_session.add(
        ReferenceDistribution(
            model_id=model_id,
            feature_name="income",
            distribution=_compute_histogram(reference_values),
            stats=_compute_distribution_stats(reference_values),
        )
    )
    db_session.add(
        Snapshot(
            model_id=model_id,
            window_date=date(2026, 3, 27),
            feature_name="income",
            distribution=_compute_histogram(production_values),
            stats=_compute_distribution_stats(production_values),
            sample_count=len(production_values),
        )
    )
    await db_session.commit()

    await run_drift_analysis(
        db=db_session, model_id=model_id, window_date=date(2026, 3, 27)
    )

    result = await db_session.execute(select(Alert).where(Alert.model_id == model_id))
    alerts = result.scalars().all()

    assert len(alerts) >= 1
    assert any(item.alert_type == "drift_red" for item in alerts)


@pytest.mark.asyncio
async def test_dispatch_webhook_called_when_url_configured(
    db_session: AsyncSession,
) -> None:
    model_id = await _create_model(db_session)
    alert = Alert(
        model_id=model_id,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift",
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)

    with patch(
        "app.services.alert_service.settings.alert_webhook_url",
        "https://example.com/webhook",
    ):
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as post_mock:
            await _dispatch_alert(alert)

    post_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_no_crash_when_webhook_fails(db_session: AsyncSession) -> None:
    model_id = await _create_model(db_session)
    alert = Alert(
        model_id=model_id,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift",
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)

    with patch(
        "app.services.alert_service.settings.alert_webhook_url",
        "https://example.com/webhook",
    ):
        with patch(
            "httpx.AsyncClient.post", new=AsyncMock(side_effect=Exception("boom"))
        ):
            await _dispatch_alert(alert)


@pytest.mark.asyncio
async def test_dispatch_silent_when_no_url(db_session: AsyncSession) -> None:
    model_id = await _create_model(db_session)
    alert = Alert(
        model_id=model_id,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift",
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)

    with patch("app.services.alert_service.settings.alert_webhook_url", ""):
        with patch("httpx.AsyncClient.post", new=AsyncMock()) as post_mock:
            await _dispatch_alert(alert)

    post_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_alert_sets_resolved_at(db_session: AsyncSession) -> None:
    model_id = await _create_model(db_session)

    alert = await create_alert(
        db=db_session,
        model_id=model_id,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift detected",
    )

    resolved = await resolve_alert(db=db_session, alert_id=alert.id)

    assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_resolve_alert_not_found_returns_404(db_session: AsyncSession) -> None:
    with pytest.raises(HTTPException) as exc_info:
        await resolve_alert(
            db=db_session, alert_id=UUID("00000000-0000-0000-0000-000000000001")
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_alerts_filters_resolved(db_session: AsyncSession) -> None:
    model_id = await _create_model(db_session)

    unresolved = await create_alert(
        db=db_session,
        model_id=model_id,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Unresolved alert",
    )
    resolved = await create_alert(
        db=db_session,
        model_id=model_id,
        feature_name="credit_score",
        alert_type="drift_red",
        severity="red",
        message="Resolved alert",
    )
    await resolve_alert(db=db_session, alert_id=resolved.id)

    unresolved_items = await list_alerts(
        db=db_session, model_id=model_id, resolved=False
    )
    resolved_items = await list_alerts(db=db_session, model_id=model_id, resolved=True)

    assert any(item.id == unresolved.id for item in unresolved_items)
    assert all(item.resolved_at is None for item in unresolved_items)
    assert any(item.id == resolved.id for item in resolved_items)
    assert all(item.resolved_at is not None for item in resolved_items)
