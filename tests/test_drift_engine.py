from __future__ import annotations

import os
from datetime import date
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.drift_score import DriftScore
from app.models.feature_importance import FeatureImportance
from app.models.model_registry import ModelRegistry
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot
from app.services.drift_engine import (
    _resolve_feature_weights,
    classify_severity,
    compute_js_divergence,
    compute_ks_test,
    compute_psi,
    run_drift_analysis,
)
from app.services.model_service import _compute_distribution_stats, _compute_histogram


def test_compute_psi_no_drift() -> None:
    reference = np.linspace(0.0, 1.0, 1000)
    production = np.linspace(0.0, 1.0, 1000)

    psi = compute_psi(reference, production)

    assert psi == pytest.approx(0.0, abs=1e-9)


def test_compute_psi_high_drift() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(0.0, 1.0, 5000)
    production = rng.normal(3.0, 1.0, 5000)

    psi = compute_psi(reference, production)

    assert psi > 0.20


def test_compute_psi_epsilon_prevents_log_zero() -> None:
    reference = np.asarray([0.0] * 50 + [1.0] * 50)
    production = np.asarray([10.0] * 100)

    psi = compute_psi(reference, production, bins=10)

    assert np.isfinite(psi)
    assert 0.0 <= psi <= 1.0


def test_compute_ks_test_significant() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(0.0, 1.0, 2000)
    production = rng.normal(3.0, 1.0, 2000)

    result = compute_ks_test(reference, production)

    assert result["significant"] is True
    assert float(result["pvalue"]) < 0.05


def test_compute_ks_test_not_significant() -> None:
    rng = np.random.default_rng(11)
    reference = rng.normal(0.0, 1.0, 2000)
    production = rng.normal(0.0, 1.0, 2000)

    result = compute_ks_test(reference, production)

    assert result["significant"] is False


def test_compute_js_divergence_identical() -> None:
    values = np.linspace(-2.0, 2.0, 1000)

    js = compute_js_divergence(values, values)

    assert js == pytest.approx(0.0, abs=1e-9)


def test_compute_js_divergence_different() -> None:
    rng = np.random.default_rng(13)
    reference = rng.normal(0.0, 1.0, 4000)
    production = rng.normal(2.5, 1.0, 4000)

    js = compute_js_divergence(reference, production)

    assert 0.0 < js <= 1.0


def test_classify_severity_all_bands() -> None:
    assert classify_severity(0.05) == "green"
    assert classify_severity(0.10) == "yellow"
    assert classify_severity(0.20) == "yellow"
    assert classify_severity(0.21) == "red"


def test_weighted_score_with_importances() -> None:
    weights = _resolve_feature_weights(
        feature_names=["income", "credit_score"],
        importances={"income": 0.9, "credit_score": 0.1},
    )

    income_weighted = 0.4 * weights["income"]
    credit_weighted = 0.4 * weights["credit_score"]

    assert income_weighted > credit_weighted


def test_weighted_score_equal_weight_fallback() -> None:
    weights = _resolve_feature_weights(
        feature_names=["f1", "f2", "f3", "f4"],
        importances={},
    )

    assert weights == {
        "f1": pytest.approx(0.25),
        "f2": pytest.approx(0.25),
        "f3": pytest.approx(0.25),
        "f4": pytest.approx(0.25),
    }


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for drift engine integration checks."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provide isolated async DB session for integration checks."""
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
async def test_run_drift_analysis_writes_rows(db_session: AsyncSession) -> None:
    model = ModelRegistry(name=f"drift-model-{uuid4()}", version="v1")
    db_session.add(model)
    await db_session.flush()

    reference_values = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
    production_values = [8.0, 9.0, 10.0, 11.0, 12.0] * 20

    db_session.add(
        ReferenceDistribution(
            model_id=model.id,
            feature_name="income",
            distribution=_compute_histogram(reference_values),
            stats=_compute_distribution_stats(reference_values),
        )
    )
    db_session.add(
        Snapshot(
            model_id=model.id,
            window_date=date(2026, 3, 27),
            feature_name="income",
            distribution=_compute_histogram(production_values),
            stats=_compute_distribution_stats(production_values),
            sample_count=len(production_values),
        )
    )
    db_session.add(
        FeatureImportance(
            model_id=model.id,
            feature_name="income",
            importance=1.0,
            method="manual",
        )
    )

    await db_session.commit()

    results = await run_drift_analysis(
        db=db_session,
        model_id=model.id,
        window_date=date(2026, 3, 27),
    )

    assert len(results) == 1

    persisted = await db_session.execute(
        select(DriftScore).where(
            DriftScore.model_id == model.id,
            DriftScore.window_date == date(2026, 3, 27),
            DriftScore.feature_name == "income",
        )
    )
    row = persisted.scalar_one_or_none()

    assert row is not None
    assert row.psi is not None
    assert row.severity in {"green", "yellow", "red"}
