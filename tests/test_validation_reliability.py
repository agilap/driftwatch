"""Regression coverage for invalid batches and immutable reference baselines."""

import asyncio
import os
from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.drift_score import DriftScore
from app.models.health_report import HealthReport
from app.models.model_registry import ModelRegistry
from app.models.snapshot import Snapshot
from app.schemas.ingest import (
    ReferencePayload,
    SnapshotPayload,
    ImportanceUploadRequest,
)
from app.schemas.model import ModelCreate
from app.services.drift_engine import run_drift_analysis
from app.services.report_generator import generate_report
from app.services.reference_service import register_reference
from app.services.snapshot_service import ingest_snapshot
from app.services.importance_scorer import normalise_importances


@pytest.mark.parametrize(
    "features",
    [
        {},
        {"x": []},
        {" ": [1]},
        {"x": [float("nan")]},
        {"x": [float("inf")]},
        {"x": [1], "y": [1, 2]},
    ],
)
def test_reject_invalid_features(features):
    for schema in (ReferencePayload, SnapshotPayload):
        with pytest.raises(ValidationError):
            schema(
                model_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                features=features,
            )


@pytest.mark.parametrize("predictions", [[], [1, 2], [float("nan")]])
def test_reject_invalid_predictions(predictions):
    for schema in (ReferencePayload, SnapshotPayload):
        with pytest.raises(ValidationError):
            schema(
                model_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                features={"x": [1]},
                predictions=predictions,
            )


def test_reserved_prediction_name_and_blank_model():
    for schema in (ReferencePayload, SnapshotPayload):
        with pytest.raises(ValidationError):
            schema(
                model_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                features={"__predictions__": [1]},
            )
    with pytest.raises(ValidationError):
        ModelCreate(name="  ")


@pytest.mark.parametrize(
    "weights",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": -1},
        {"x": 0},
        {"x": 1e308, "y": 1e308},
    ],
)
def test_reject_invalid_importances(weights):
    with pytest.raises(ValidationError):
        ImportanceUploadRequest(importances=weights, method="manual")
    with pytest.raises(ValueError):
        normalise_importances(weights)


def test_reference_importances_must_match_features():
    with pytest.raises(ValidationError):
        ReferencePayload(
            model_id=uuid4(), features={"x": [1]}, feature_importances={"y": 1}
        )


@pytest.mark.asyncio
async def test_reference_membership_freeze_and_concurrent_ingestion():
    url = os.environ["DATABASE_URL"]
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    # Alembic uses asyncio.run, so run the synchronous migration in a thread.
    await asyncio.to_thread(command.upgrade, cfg, "head")
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    model_id = uuid4()
    try:
        async with sessions() as db:
            db.add(ModelRegistry(id=model_id, name=f"validation-{model_id}"))
            await db.commit()
        payload = SnapshotPayload(
            model_id=model_id,
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            features={"x": [1, 2, 3]},
        )
        async with sessions() as db:
            with pytest.raises(HTTPException) as missing:
                await ingest_snapshot(db, payload)
            assert missing.value.status_code == 409
        reference = ReferencePayload(model_id=model_id, features={"x": [1, 2, 3]})
        async with sessions() as db:
            await register_reference(db, reference)
        async with sessions() as db:
            with pytest.raises(HTTPException) as unknown:
                await ingest_snapshot(
                    db, payload.model_copy(update={"features": {"other": [1, 2, 3]}})
                )
            assert unknown.value.status_code == 422

        async def ingest():
            async with sessions() as db:
                await ingest_snapshot(db, payload)

        await asyncio.gather(ingest(), ingest())
        async with sessions() as db:
            count = await db.scalar(
                select(func.count())
                .select_from(Snapshot)
                .where(Snapshot.model_id == model_id)
            )
            assert count == 1
            await register_reference(db, reference)
            with pytest.raises(HTTPException) as frozen:
                await register_reference(
                    db,
                    ReferencePayload(model_id=model_id, features={"x": [10, 20, 30]}),
                )
            assert frozen.value.status_code == 409
        async with sessions() as db:
            await run_drift_analysis(db, model_id, date(2026, 1, 1))
            report = await generate_report(
                db, model_id, date(2025, 12, 29), date(2026, 1, 4)
            )
            report_id = report.id
            scores = (
                (
                    await db.execute(
                        select(DriftScore).where(DriftScore.model_id == model_id)
                    )
                )
                .scalars()
                .all()
            )
            assert len(scores) == 1
            assert scores[0].psi == pytest.approx(0)
        # Identical retries preserve evidence; changed samples invalidate it in
        # the ingestion transaction, even if the background worker never runs.
        async with sessions() as db:
            await ingest_snapshot(db, payload)
            assert await db.get(HealthReport, report_id) is not None
        async with sessions() as db:
            await ingest_snapshot(
                db, payload.model_copy(update={"features": {"x": [10, 20, 30]}})
            )
        async with sessions() as db:
            assert await db.get(HealthReport, report_id) is None
            remaining_scores = await db.scalar(
                select(func.count())
                .select_from(DriftScore)
                .where(DriftScore.model_id == model_id)
            )
            assert remaining_scores == 0
            snapshot = (
                await db.execute(select(Snapshot).where(Snapshot.model_id == model_id))
            ).scalar_one()
            assert snapshot.stats["mean"] == pytest.approx(20)
    finally:
        async with sessions() as db:
            model = await db.get(ModelRegistry, model_id)
            if model:
                await db.delete(model)
                await db.commit()
        await engine.dispose()


@pytest.mark.asyncio
async def test_prediction_baseline_scoring_and_alerts():
    from app.models.alert import Alert
    from app.models.reference_distribution import ReferenceDistribution

    url = os.environ["DATABASE_URL"]
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    await asyncio.to_thread(command.upgrade, cfg, "head")
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    model_ids = [uuid4(), uuid4()]
    values = [float(i % 10) for i in range(200)]
    predictions = [v / 10 for v in values]
    try:
        async with sessions() as db:
            db.add_all([ModelRegistry(id=i, name=f"prediction-{i}") for i in model_ids])
            await db.commit()
            reference = ReferencePayload(
                model_id=model_ids[0], features={"x": values}, predictions=predictions
            )
            await register_reference(db, reference)
            await register_reference(
                db, ReferencePayload(model_id=model_ids[1], features={"x": values})
            )
            for model_id in model_ids:
                await ingest_snapshot(
                    db,
                    SnapshotPayload(
                        model_id=model_id,
                        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        features={"x": [v + 3 for v in values]},
                        predictions=[2.0] * len(values),
                    ),
                )
            await register_reference(db, reference)
            for model_id, baseline in (
                (model_ids[0], [0.5] * 200),
                (model_ids[1], predictions),
            ):
                with pytest.raises(HTTPException) as frozen:
                    await register_reference(
                        db,
                        ReferencePayload(
                            model_id=model_id,
                            features={"x": values},
                            predictions=baseline,
                        ),
                    )
                assert frozen.value.status_code == 409
            ref = (
                await db.execute(
                    select(ReferenceDistribution).where(
                        ReferenceDistribution.model_id == model_ids[0],
                        ReferenceDistribution.feature_name == "__predictions__",
                    )
                )
            ).scalar_one()
            snap = (
                await db.execute(
                    select(Snapshot).where(
                        Snapshot.model_id == model_ids[0],
                        Snapshot.feature_name == "__predictions__",
                    )
                )
            ).scalar_one()
            assert ref.distribution["bin_edges"] == snap.distribution["bin_edges"]
            scored = await run_drift_analysis(db, model_ids[0], date(2026, 1, 1))
            without_baseline = await run_drift_analysis(
                db, model_ids[1], date(2026, 1, 1)
            )
            assert {row.feature_name for row in without_baseline} == {"x"}
            output = next(
                row for row in scored if row.feature_name == "__predictions__"
            )
            assert output.severity == "red"
            assert (
                output.psi > 0 and output.js_divergence > 0 and output.ks_statistic > 0
            )
            assert output.weighted_score == 0
            assert next(
                row for row in scored if row.feature_name == "x"
            ).weighted_score == pytest.approx(without_baseline[0].weighted_score)
            report = await generate_report(
                db, model_ids[0], date(2025, 12, 29), date(2026, 1, 4)
            )
            other_report = await generate_report(
                db, model_ids[1], date(2025, 12, 29), date(2026, 1, 4)
            )
            assert report.overall_score == other_report.overall_score
            assert report.report_json["output_distribution"][
                "output_drift_psi"
            ] == pytest.approx(output.psi)
            stored = (
                await db.execute(
                    select(DriftScore).where(
                        DriftScore.model_id == model_ids[0],
                        DriftScore.feature_name == "x",
                    )
                )
            ).scalar_one()
            assert stored.weighted_score > 0
            assert report.report_json["feature_drift_summary"][0][
                "weighted_score"
            ] == pytest.approx(stored.weighted_score)
            await run_drift_analysis(db, model_ids[0], date(2026, 1, 1))
            alerts = (
                (
                    await db.execute(
                        select(Alert).where(
                            Alert.model_id == model_ids[0],
                            Alert.feature_name == "__predictions__",
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert len(alerts) == 1
            assert alerts[0].alert_type == "output_shift"
            assert alerts[0].message.startswith("Model prediction distribution PSI=")
    finally:
        async with sessions() as db:
            for model_id in model_ids:
                model = await db.get(ModelRegistry, model_id)
                if model:
                    await db.delete(model)
            await db.commit()
        await engine.dispose()
