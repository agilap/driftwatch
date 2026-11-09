from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from sqlalchemy import delete, select

from app.models.alert import Alert
from app.models.drift_score import DriftScore
from app.models.feature_importance import FeatureImportance
from app.models.snapshot import Snapshot
from app.schemas.ingest import SnapshotPayload
from app.services.importance_scorer import upsert_importances
from app.services.snapshot_service import ingest_snapshot
from tests.test_reliability_fixes import (
    ctx,  # noqa: F401
    _model,
    _reference,
    run_migrations,  # noqa: F401
)


@pytest.mark.asyncio
async def test_importances_return_after_concurrent_deletion(
    ctx, monkeypatch  # noqa: F811
):  # noqa: F811
    client, factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    deleted = []
    async with factory() as session:
        original_commit = session.commit

        async def commit_and_delete():
            await original_commit()
            async with factory() as other:
                result = await other.execute(
                    delete(FeatureImportance)
                    .where(FeatureImportance.model_id == model_id)
                    .returning(FeatureImportance.id)
                )
                deleted.extend(result.scalars().all())
                await other.commit()

        monkeypatch.setattr(session, "commit", commit_and_delete)
        rows = await upsert_importances(
            session, model_id, {"a": 0.5, "b": 0.5}, "manual"
        )
    assert set(deleted) == {row.id for row in rows}
    assert [row.feature_name for row in rows] == ["a", "b"]


@pytest.mark.asyncio
async def test_stats_only_snapshot_change_invalidates_evidence(ctx):  # noqa: F811
    client, factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    timestamp = "2026-02-02T12:00:00Z"
    body = {
        "model_id": model_id,
        "timestamp": timestamp,
        "features": {"a": [10.0] * 200},
    }
    with patch("app.services.alert_service._dispatch_alert", new=AsyncMock()):
        response = await client.post("/ingest/snapshot", json=body)
    assert response.status_code == 200, response.text
    async with factory() as session:
        snapshot = await session.scalar(
            select(Snapshot).where(Snapshot.model_id == model_id)
        )
        distribution, mean = snapshot.distribution, snapshot.stats["mean"]
        alert = await session.scalar(select(Alert).where(Alert.model_id == model_id))
        assert alert is not None
        alert_id = alert.id
        assert (
            await session.scalar(
                select(DriftScore.id).where(DriftScore.model_id == model_id)
            )
            is not None
        )
        await ingest_snapshot(
            session,
            SnapshotPayload(
                model_id=UUID(model_id),
                timestamp=datetime(2026, 2, 2, 12, tzinfo=timezone.utc),
                features={"a": [11.0] * 200},
            ),
        )
    async with factory() as session:
        snapshot = await session.scalar(
            select(Snapshot).where(Snapshot.model_id == model_id)
        )
        assert snapshot.distribution == distribution
        assert snapshot.stats["mean"] != mean
        alert = await session.get(Alert, alert_id)
        assert alert.superseded_at is not None
        assert (
            await session.scalar(
                select(DriftScore.id).where(DriftScore.model_id == model_id)
            )
            is None
        )


@pytest.mark.asyncio
async def test_analyze_without_reference_returns_conflict(ctx):  # noqa: F811
    client, _ = ctx
    model_id = await _model(client)
    response = await client.post(f"/drift/{model_id}/{date(2026, 2, 2)}/analyze")
    assert response.status_code == 409
    assert response.json()["detail"] == "Reference distribution required"
