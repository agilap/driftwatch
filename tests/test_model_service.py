from __future__ import annotations

import os
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.services.model_service import (
    _compute_distribution_stats,
    _compute_histogram,
    create_model,
)


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for DB-backed service tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provide isolated async DB session for each test."""
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


@pytest.mark.asyncio
async def test_create_model_success(db_session: AsyncSession) -> None:
    created = await create_model(db=db_session, name=f"loan-scorer-{uuid4()}", version="v1")

    assert isinstance(created.id, UUID)
    assert created.name.startswith("loan-scorer-")
    assert created.version == "v1"


@pytest.mark.asyncio
async def test_create_model_duplicate_name_fails(db_session: AsyncSession) -> None:
    name = f"shared-model-{uuid4()}"
    await create_model(db=db_session, name=name, version="v1")

    with pytest.raises(IntegrityError):
        await create_model(db=db_session, name=name, version="v2")


def test_compute_distribution_stats_basic() -> None:
    stats = _compute_distribution_stats([1.0, 2.0, 3.0, 4.0])

    assert stats["mean"] == pytest.approx(2.5)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)
    assert stats["p50"] == pytest.approx(2.5)


def test_compute_histogram_bin_count() -> None:
    histogram = _compute_histogram([1.0, 2.0, 3.0, 4.0, 5.0], bins=10)

    assert len(histogram["counts"]) == 10
    assert len(histogram["bins"]) == 10
    assert len(histogram["bin_edges"]) == 11
    assert sum(histogram["counts"]) == 5
