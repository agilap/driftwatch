from datetime import date, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from app.config import settings
from app.models.drift_score import DriftScore
from app.models.model_registry import ModelRegistry
from app.schemas.ingest import PREDICTIONS_FEATURE_NAME, SnapshotPayload
from app.services.model_service import _compute_histogram
from app.services.report_generator import generate_report
from app.services.snapshot_service import ingest_snapshot
from tests.helpers import _model, _reference


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["ingest", "analysis"])
@pytest.mark.parametrize(
    "day, expected_status",
    [("2026-02-02", 404), ("2026-02-08", 404), ("2026-02-09", 200)],
)
async def test_report_invalidation_week_boundaries(ctx, path, day, expected_status):
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    payload = SnapshotPayload(
        model_id=UUID(model_id),
        timestamp=datetime.fromisoformat(f"{day}T12:00:00+00:00"),
        features={"a": [10.0] * 200},
    )
    if path == "analysis":
        async with session_factory() as session:
            await ingest_snapshot(session, payload)
    report_url = f"/reports/{model_id}/2026-02-02"
    response = await client.post(
        f"/reports/{model_id}/generate", json={"week_start": "2026-02-02"}
    )
    assert response.status_code == 200, response.text
    assert (await client.get(report_url)).status_code == 200
    if path == "ingest":
        async with session_factory() as session:
            await ingest_snapshot(session, payload)
    else:
        with patch("app.services.alert_service._dispatch_alert", new=AsyncMock()):
            response = await client.post(f"/drift/{model_id}/{day}/analyze")
        assert response.status_code == 200, response.text
        assert response.json()
    assert (await client.get(report_url)).status_code == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timestamp, status",
    [
        ("2026-03-02T01:00:00+08:00", 422),
        ("2026-03-02T01:00:00Z", 200),
        ("2026-03-02T01:00:00+00:00", 200),
    ],
)
async def test_snapshot_requires_utc_timestamp(ctx, timestamp, status):
    client, _ = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": model_id,
            "timestamp": timestamp,
            "features": {"a": [1.0, 2.0]},
        },
    )
    assert response.status_code == status, response.text
    dates = await client.get(f"/ingest/snapshots/{model_id}")
    assert dates.status_code == 200
    assert dates.json() == ([] if status == 422 else ["2026-03-02"])
    if status == 200:
        assert response.json()["window_date"] == "2026-03-02"


@pytest.mark.asyncio
async def test_frozen_reference_rejects_changed_stats_with_same_histogram(ctx):
    client, _ = ctx
    model_id = await _model(client)
    original = [0.0, 1.1, 5.1, 10.0]
    changed = [0.0, 1.9, 5.9, 10.0]
    assert _compute_histogram(original) == _compute_histogram(changed)
    assert sum(original) != sum(changed)
    body = {"model_id": model_id, "features": {"a": original}}
    response = await client.post("/ingest/reference", json=body)
    assert response.status_code == 200, response.text
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": model_id,
            "timestamp": "2026-03-02T01:00:00Z",
            "features": {"a": [1.0, 2.0]},
        },
    )
    assert response.status_code == 200, response.text
    before = await client.get(f"/ingest/reference/{model_id}")
    assert before.status_code == 200
    response = await client.post(
        "/ingest/reference", json={"model_id": model_id, "features": {"a": changed}}
    )
    assert response.status_code == 409, response.text
    after = await client.get(f"/ingest/reference/{model_id}")
    assert after.status_code == 200
    assert after.json() == before.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("score, expected", [(60.0, False), (59.99, True)])
async def test_health_recommendation_cutoff(db_session, monkeypatch, score, expected):
    model = ModelRegistry(name=f"boundary-{uuid4()}", version="v1")
    db_session.add(model)
    await db_session.commit()
    monkeypatch.setattr(
        "app.services.report_generator.compute_overall_health_score", lambda _: score
    )
    report = await generate_report(
        db_session, model.id, date(2026, 3, 16), date(2026, 3, 22)
    )
    assert (
        any(
            "Evaluate model performance" in item
            for item in report.report_json["recommendations"]
        )
        is expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("above, expected", [(False, False), (True, True)])
async def test_output_drift_recommendation_cutoff(db_session, above, expected):
    model = ModelRegistry(name=f"boundary-{uuid4()}", version="v1")
    db_session.add(model)
    await db_session.flush()
    db_session.add(
        DriftScore(
            model_id=model.id,
            window_date=date(2026, 3, 16),
            feature_name=PREDICTIONS_FEATURE_NAME,
            ks_statistic=0.1,
            ks_pvalue=0.04,
            psi=settings.psi_red_threshold + (0.01 if above else 0.0),
            js_divergence=0.0,
            weighted_score=0.0,
            severity="green",
        )
    )
    await db_session.commit()
    report = await generate_report(
        db_session, model.id, date(2026, 3, 16), date(2026, 3, 22)
    )
    assert (
        "Output distribution has significantly shifted"
        in report.report_json["recommendations"]
    ) is expected
