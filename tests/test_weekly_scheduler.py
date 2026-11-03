"""The weekly report job against a real database."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.health_report import HealthReport
from app.models.model_registry import ModelRegistry
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot
from app.services.drift_engine import run_drift_analysis
from app.services.model_service import _compute_distribution_stats, _compute_histogram
from scheduler import weekly_report
from scheduler.weekly_report import run_weekly_reports_job


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def sessions(monkeypatch):
    engine = create_async_engine(os.environ["DATABASE_URL"])
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(
            text(
                "TRUNCATE TABLE "
                "alerts, health_reports, drift_scores, snapshots, "
                "feature_importances, reference_distributions, models "
                "RESTART IDENTITY CASCADE"
            )
        )
        await session.commit()
    monkeypatch.setattr(weekly_report, "AsyncSessionLocal", factory)
    yield factory
    await engine.dispose()


async def _model(sessions, name: str, with_window: bool = False) -> ModelRegistry:
    values = [1.0, 2.0, 3.0, 4.0] * 50
    window = datetime.now(timezone.utc).date() - timedelta(days=1)
    async with sessions() as session:
        model = ModelRegistry(name=name)
        session.add(model)
        await session.flush()
        if with_window:
            histogram = _compute_histogram(values)
            stats = _compute_distribution_stats(values)
            session.add(
                ReferenceDistribution(
                    model_id=model.id,
                    feature_name="a",
                    distribution=histogram,
                    stats=stats,
                )
            )
            session.add(
                Snapshot(
                    model_id=model.id,
                    window_date=window,
                    feature_name="a",
                    distribution=histogram,
                    stats=stats,
                    sample_count=len(values),
                )
            )
        await session.commit()
        if with_window:
            await run_drift_analysis(session, model.id, window)
        return model


async def _scores(sessions) -> dict:
    async with sessions() as session:
        rows = (
            await session.execute(
                select(ModelRegistry.name, HealthReport.overall_score).join(
                    HealthReport, HealthReport.model_id == ModelRegistry.id
                )
            )
        ).all()
    return {row.name: row.overall_score for row in rows}


@pytest.mark.asyncio
async def test_job_scores_data_windows_and_nulls_empty_models(sessions) -> None:
    await _model(sessions, "with-data", with_window=True)
    await _model(sessions, "without-data")
    await run_weekly_reports_job()
    scores = await _scores(sessions)
    assert scores["with-data"] == pytest.approx(100.0)
    assert scores["without-data"] is None


@pytest.mark.asyncio
async def test_one_model_failing_does_not_stop_the_others(
    sessions, monkeypatch, caplog
) -> None:
    failing = await _model(sessions, "a-fails", with_window=True)
    await _model(sessions, "b-works", with_window=True)
    real = weekly_report.generate_report

    async def _generate(db, model_id, week_start, week_end):
        if model_id == failing.id:
            raise RuntimeError("report failed")
        return await real(
            db=db, model_id=model_id, week_start=week_start, week_end=week_end
        )

    monkeypatch.setattr(weekly_report, "generate_report", _generate)
    await run_weekly_reports_job()
    assert set(await _scores(sessions)) == {"b-works"}
    assert "Weekly report failed" in caplog.text
