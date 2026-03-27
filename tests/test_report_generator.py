from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import date
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.drift_score import DriftScore
from app.models.health_report import HealthReport
from app.models.model_registry import ModelRegistry
from app.services.report_generator import compute_overall_health_score, generate_report


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for report generator tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide isolated async DB session for report tests."""
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


async def _create_model(db: AsyncSession, name: str | None = None) -> ModelRegistry:
    model = ModelRegistry(name=name or f"report-model-{uuid4()}", version="v1")
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


async def _insert_week_drift(
    db: AsyncSession,
    model_id,
    week_start: date,
    feature_rows: list[dict],
) -> None:
    for day_offset in range(7):
        window_date = week_start.fromordinal(week_start.toordinal() + day_offset)
        for row in feature_rows:
            db.add(
                DriftScore(
                    model_id=model_id,
                    window_date=window_date,
                    feature_name=row["feature_name"],
                    ks_statistic=row.get("ks_statistic", 0.1),
                    ks_pvalue=row.get("ks_pvalue", 0.04),
                    psi=row.get("psi", 0.0),
                    js_divergence=row.get("js_divergence", 0.0),
                    weighted_score=row.get("weighted_score", 0.0),
                    severity=row.get("severity", "green"),
                )
            )
    await db.commit()


def test_overall_health_score_no_drift() -> None:
    score = compute_overall_health_score([0.0, 0.0, 0.0])
    assert score == pytest.approx(100.0)


def test_overall_health_score_full_drift() -> None:
    score = compute_overall_health_score([1.0, 1.0, 1.0])
    assert score == pytest.approx(0.0)


def test_overall_health_score_clamped_to_range() -> None:
    assert compute_overall_health_score([10.0]) == pytest.approx(0.0)
    assert compute_overall_health_score([-10.0]) == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_recommendation_retraining_when_score_below_60(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [{"feature_name": "income", "psi": 1.0, "severity": "red"}],
    )

    report = await generate_report(db_session, model.id, week_start, week_end)
    recommendations = report.report_json["recommendations"]

    assert any("Model retraining recommended" in item for item in recommendations)


@pytest.mark.asyncio
async def test_recommendation_red_feature_flagged(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [{"feature_name": "credit_score", "psi": 0.35, "severity": "red"}],
    )

    report = await generate_report(db_session, model.id, week_start, week_end)
    recommendations = report.report_json["recommendations"]

    assert any("Investigate credit_score immediately" in item for item in recommendations)


@pytest.mark.asyncio
async def test_recommendation_output_shift_flagged(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [
            {"feature_name": "income", "psi": 0.05, "severity": "green"},
            {"feature_name": "__predictions__", "psi": 0.25, "severity": "red"},
        ],
    )

    report = await generate_report(db_session, model.id, week_start, week_end)
    recommendations = report.report_json["recommendations"]

    assert any("Output distribution has significantly shifted" in item for item in recommendations)


@pytest.mark.asyncio
async def test_generate_report_structure(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [
            {"feature_name": "income", "psi": 0.12, "severity": "yellow"},
            {"feature_name": "age", "psi": 0.05, "severity": "green"},
            {"feature_name": "__predictions__", "psi": 0.09, "severity": "green"},
        ],
    )

    report = await generate_report(db_session, model.id, week_start, week_end)
    report_json = report.report_json

    assert "executive_summary" in report_json
    assert "feature_drift_summary" in report_json
    assert "output_distribution" in report_json
    assert "top_drifted_features" in report_json
    assert "feature_to_output_correlation" in report_json
    assert "recommendations" in report_json


@pytest.mark.asyncio
async def test_generate_report_markdown_renders(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [{"feature_name": "income", "psi": 0.11, "severity": "yellow"}],
    )

    report = await generate_report(db_session, model.id, week_start, week_end)
    markdown = report.report_markdown or ""

    assert markdown
    assert "# DriftWatch Weekly Health Report" in markdown
    assert "## Executive Summary" in markdown
    assert "## Feature Drift Summary" in markdown
    assert "## Output Distribution" in markdown
    assert "## Feature to Output Correlation" in markdown
    assert "## Recommendations" in markdown


@pytest.mark.asyncio
async def test_generate_report_stores_to_db(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [{"feature_name": "income", "psi": 0.1, "severity": "yellow"}],
    )

    report = await generate_report(db_session, model.id, week_start, week_end)

    persisted = await db_session.execute(select(HealthReport).where(HealthReport.id == report.id))
    row = persisted.scalar_one_or_none()

    assert row is not None
    assert row.model_id == model.id
    assert row.week_start == week_start


@pytest.mark.asyncio
async def test_generate_report_upsert(db_session: AsyncSession) -> None:
    model = await _create_model(db_session)
    week_start = date(2026, 3, 16)
    week_end = date(2026, 3, 22)

    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [{"feature_name": "income", "psi": 0.2, "severity": "yellow"}],
    )

    first = await generate_report(db_session, model.id, week_start, week_end)

    # Add more severe drift and regenerate for same week_start.
    await db_session.execute(text("DELETE FROM drift_scores WHERE model_id = :model_id"), {"model_id": str(model.id)})
    await db_session.commit()
    await _insert_week_drift(
        db_session,
        model.id,
        week_start,
        [{"feature_name": "income", "psi": 0.9, "severity": "red"}],
    )

    second = await generate_report(db_session, model.id, week_start, week_end)

    rows_result = await db_session.execute(
        select(HealthReport).where(
            HealthReport.model_id == model.id,
            HealthReport.week_start == week_start,
        )
    )
    rows = rows_result.scalars().all()

    assert len(rows) == 1
    assert first.id == second.id
    assert second.overall_score <= first.overall_score
