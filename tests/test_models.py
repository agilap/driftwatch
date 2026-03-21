from __future__ import annotations

import os
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.feature_importance import FeatureImportance
from app.models.model_registry import ModelRegistry
from app.models.reference_distribution import ReferenceDistribution


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply schema migrations before model tests run."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provide an async DB session bound to the migrated test database."""
    engine = create_async_engine(_database_url(), future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_all_schema_tables_exist() -> None:
    engine = create_async_engine(_database_url(), future=True)
    async with engine.connect() as conn:
        table_names = await conn.run_sync(
            lambda sync_conn: set(inspect(sync_conn).get_table_names())
        )
    await engine.dispose()

    expected_tables = {
        "models",
        "reference_distributions",
        "feature_importances",
        "snapshots",
        "drift_scores",
        "health_reports",
        "alerts",
    }
    assert expected_tables.issubset(table_names)


@pytest.mark.asyncio
async def test_fk_constraints_enforced(db_session: AsyncSession) -> None:
    orphan_reference = ReferenceDistribution(
        model_id=uuid4(),
        feature_name="income",
        distribution={"bins": [0, 1], "counts": [10]},
        stats={
            "mean": 0.5,
            "std": 0.2,
            "min": 0.0,
            "max": 1.0,
            "p25": 0.25,
            "p50": 0.5,
            "p75": 0.75,
        },
    )
    db_session.add(orphan_reference)

    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()


@pytest.mark.asyncio
async def test_unique_constraints_reference_and_importance(db_session: AsyncSession) -> None:
    model = ModelRegistry(name=f"model-{uuid4()}", version="v1")
    db_session.add(model)
    await db_session.flush()

    ref_one = ReferenceDistribution(
        model_id=model.id,
        feature_name="credit_score",
        distribution={"bins": [0, 1], "counts": [20]},
        stats={"mean": 0.6, "std": 0.1, "min": 0.0, "max": 1.0, "p25": 0.3, "p50": 0.6, "p75": 0.8},
    )
    ref_two = ReferenceDistribution(
        model_id=model.id,
        feature_name="credit_score",
        distribution={"bins": [0, 1], "counts": [30]},
        stats={"mean": 0.7, "std": 0.1, "min": 0.0, "max": 1.0, "p25": 0.3, "p50": 0.6, "p75": 0.8},
    )
    db_session.add_all([ref_one, ref_two])

    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()

    model_two = ModelRegistry(name=f"model-{uuid4()}", version="v1")
    db_session.add(model_two)
    await db_session.flush()

    imp_one = FeatureImportance(
        model_id=model_two.id,
        feature_name="income",
        importance=0.42,
        method="manual",
    )
    imp_two = FeatureImportance(
        model_id=model_two.id,
        feature_name="income",
        importance=0.51,
        method="shap",
    )
    db_session.add_all([imp_one, imp_two])

    with pytest.raises(IntegrityError):
        await db_session.flush()
    await db_session.rollback()
