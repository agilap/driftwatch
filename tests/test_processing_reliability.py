from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.routers.drift import router
from app.routers.ingest import analyze_snapshot_in_background
from app.services.alert_service import create_alert


# Reuse the PostgreSQL isolation fixtures already used by the alert suite.
from tests.test_alert_service import (
    db_session as db_session,
    run_migrations as run_migrations,
)


def test_latest_drift_route_precedes_date_route():
    paths = [route.path for route in router.routes]
    assert paths.index("/{model_id}/latest") < paths.index("/{model_id}/{window_date}")


@pytest.mark.asyncio
async def test_background_analysis_owns_and_closes_session():
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    model_id = uuid4()
    window = date(2026, 1, 1)
    with patch("app.routers.ingest.AsyncSessionLocal", return_value=context), patch(
        "app.routers.ingest.run_drift_analysis", new=AsyncMock()
    ) as analyze:
        await analyze_snapshot_in_background(model_id, window)
    analyze.assert_awaited_once_with(db=session, model_id=model_id, window_date=window)
    context.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_background_failure_rolls_back_and_logs():
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch("app.routers.ingest.AsyncSessionLocal", return_value=context), patch(
        "app.routers.ingest.run_drift_analysis",
        new=AsyncMock(side_effect=RuntimeError("broken")),
    ), patch("app.routers.ingest.logger.exception") as log_failure:
        await analyze_snapshot_in_background(uuid4(), date(2026, 1, 1))
    session.rollback.assert_awaited_once()
    log_failure.assert_called_once()
    assert "Drift analysis failed" in log_failure.call_args.args[0]


@pytest.mark.asyncio
async def test_alert_participates_in_callers_transaction():
    session = AsyncMock()
    session.add = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    session.execute.return_value = result
    with patch(
        "app.services.alert_service._dispatch_alert", new=AsyncMock()
    ) as dispatch:
        await create_alert(
            db=session,
            model_id=uuid4(),
            feature_name="income",
            alert_type="drift_red",
            severity="red",
            message="shift",
            window_date=date(2026, 1, 1),
            commit=False,
        )
    session.flush.assert_awaited_once()
    session.commit.assert_not_awaited()
    dispatch.assert_not_awaited()


async def _seed_shifted_model(session):
    from app.models.model_registry import ModelRegistry
    from app.models.reference_distribution import ReferenceDistribution
    from app.models.snapshot import Snapshot
    from app.services.model_service import (
        _compute_distribution_stats,
        _compute_histogram,
    )

    model = ModelRegistry(name=f"processing-{uuid4()}", version="v1")
    session.add(model)
    await session.flush()
    for feature in ["a", "b"]:
        reference = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        shifted = [11.0, 12.0, 13.0, 14.0, 15.0] * 20
        session.add(
            ReferenceDistribution(
                model_id=model.id,
                feature_name=feature,
                distribution=_compute_histogram(reference),
                stats=_compute_distribution_stats(reference),
            )
        )
        session.add(
            Snapshot(
                model_id=model.id,
                feature_name=feature,
                window_date=date(2026, 1, 1),
                distribution=_compute_histogram(
                    shifted, bin_edges=_compute_histogram(reference)["bin_edges"]
                ),
                stats=_compute_distribution_stats(shifted),
                sample_count=100,
            )
        )
    await session.commit()
    return model.id


@pytest.mark.asyncio
async def test_failed_analysis_does_not_commit_partial_scores_or_alerts(
    db_session,  # noqa: F811
):
    from sqlalchemy import select
    from app.models.alert import Alert
    from app.models.drift_score import DriftScore
    from app.services import drift_engine

    model_id = await _seed_shifted_model(db_session)
    original = drift_engine.compute_psi_from_counts
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second feature failed")
        return original(*args, **kwargs)

    with patch(
        "app.services.drift_engine.compute_psi_from_counts", side_effect=fail_second
    ):
        with pytest.raises(RuntimeError, match="second feature"):
            await drift_engine.run_drift_analysis(
                db_session, model_id, date(2026, 1, 1)
            )
    await db_session.rollback()
    assert (
        not (await db_session.execute(select(Alert).where(Alert.model_id == model_id)))
        .scalars()
        .all()
    )
    assert (
        not (
            await db_session.execute(
                select(DriftScore).where(DriftScore.model_id == model_id)
            )
        )
        .scalars()
        .all()
    )


@pytest.mark.asyncio
async def test_concurrent_analysis_deduplicates_alerts_and_scores(
    db_session,  # noqa: F811
):
    import asyncio
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from app.models.alert import Alert
    from app.models.drift_score import DriftScore
    from app.services.drift_engine import run_drift_analysis

    model_id = await _seed_shifted_model(db_session)
    factory = async_sessionmaker(db_session.bind, expire_on_commit=False)

    async def analyze():
        async with factory() as session:
            await run_drift_analysis(session, model_id, date(2026, 1, 1))

    await asyncio.gather(analyze(), analyze())
    alerts = (
        (await db_session.execute(select(Alert).where(Alert.model_id == model_id)))
        .scalars()
        .all()
    )
    scores = (
        (
            await db_session.execute(
                select(DriftScore).where(DriftScore.model_id == model_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(alerts) == 2
    assert len(scores) == 2
    assert all(alert.window_date == date(2026, 1, 1) for alert in alerts)
