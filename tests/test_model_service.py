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


@pytest.fixture(scope="session")
def run_migrations() -> None:
    """Apply migrations for DB-backed service tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session(run_migrations: None) -> AsyncSession:
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
    created = await create_model(
        db=db_session, name=f"loan-scorer-{uuid4()}", version="v1"
    )

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


def test_compute_histogram_preserves_outliers_and_endpoints() -> None:
    histogram = _compute_histogram(
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        bin_edges=[-2.0, 0.0, 2.0],
    )

    assert histogram["bin_edges"] == [-2.0, 0.0, 2.0]
    assert histogram["counts"] == [2, 3]
    assert histogram["underflow"] == 1
    assert histogram["overflow"] == 1
    assert (
        sum(histogram["counts"]) + histogram["underflow"] + histogram["overflow"] == 7
    )


def test_compute_histogram_constant_reference_has_finite_edges() -> None:
    histogram = _compute_histogram([-5.0] * 3)

    assert histogram["underflow"] == 0
    assert histogram["overflow"] == 0
    assert sum(histogram["counts"]) == 3
    assert all(-float("inf") < edge < float("inf") for edge in histogram["bin_edges"])


@pytest.mark.parametrize("values", [[], [[1.0]], [float("nan")], [float("inf")]])
@pytest.mark.parametrize("compute", [_compute_histogram, _compute_distribution_stats])
def test_distribution_helpers_reject_invalid_values(values, compute) -> None:
    with pytest.raises(ValueError):
        compute(values)


@pytest.mark.parametrize("bins", [0, -1, 1.5, True, "10"])
def test_histogram_rejects_invalid_bin_counts(bins) -> None:
    with pytest.raises(ValueError):
        _compute_histogram([1.0], bins=bins)


@pytest.mark.parametrize(
    "edges", [[], [1.0], [0.0, 0.0], [1.0, 0.0], [[0.0, 1.0]], [0.0, float("inf")]]
)
def test_histogram_rejects_invalid_edges(edges) -> None:
    with pytest.raises(ValueError):
        _compute_histogram([1.0], bin_edges=edges)
