from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from app.models.alert import Alert
from app.services import drift_engine
from tests.helpers import _model, _normal

DAY = "2026-02-02"


async def _setup(client, monkeypatch, features):
    monkeypatch.setattr(drift_engine.settings, "psi_yellow_threshold", 0.1)
    monkeypatch.setattr(drift_engine.settings, "psi_red_threshold", 0.2)
    dispatch = AsyncMock()
    monkeypatch.setattr("app.services.alert_service._dispatch_alert", dispatch)
    model_id = await _model(client)
    reference = {name: _normal(seed=1) for name in features}
    response = await client.post(
        "/ingest/reference", json={"model_id": model_id, "features": reference}
    )
    assert response.status_code == 200, response.text
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": model_id,
            "timestamp": f"{DAY}T12:00:00Z",
            "features": features,
        },
    )
    assert response.status_code == 200, response.text
    scores = await _analyze(client, model_id)
    return model_id, scores, dispatch


async def _analyze(client, model_id):
    response = await client.post(f"/drift/{model_id}/{DAY}/analyze")
    assert response.status_code == 200, response.text
    return {row["feature_name"]: row for row in response.json()}


async def _alerts(client, model_id, resolved=False):
    response = await client.get(f"/alerts/{model_id}", params={"resolved": resolved})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_threshold_change_clears_alert_and_red_again_creates_new(
    ctx, monkeypatch
):
    client, factory = ctx
    model_id, scores, dispatch = await _setup(
        client, monkeypatch, {"a": _normal(3, seed=4)}
    )
    assert scores["a"]["severity"] == "red"
    old = (await _alerts(client, model_id))["items"][0]
    monkeypatch.setattr(drift_engine.settings, "psi_yellow_threshold", 100.0)
    monkeypatch.setattr(drift_engine.settings, "psi_red_threshold", 100.0)
    assert (await _analyze(client, model_id))["a"]["severity"] == "green"
    assert (await _alerts(client, model_id))["total"] == 0
    assert [row["id"] for row in (await _alerts(client, model_id, True))["items"]] == [
        old["id"]
    ]
    async with factory() as session:
        row = await session.get(Alert, old["id"])
        assert row.superseded_at is not None
        assert row.resolved_at == row.superseded_at
    assert dispatch.await_count == 1
    monkeypatch.setattr(drift_engine.settings, "psi_yellow_threshold", 0.1)
    monkeypatch.setattr(drift_engine.settings, "psi_red_threshold", 0.2)
    assert (await _analyze(client, model_id))["a"]["severity"] == "red"
    active = await _alerts(client, model_id)
    assert active["total"] == 1
    assert active["items"][0]["id"] != old["id"]
    async with factory() as session:
        assert (
            len(
                (
                    await session.scalars(
                        select(Alert).where(Alert.model_id == model_id)
                    )
                ).all()
            )
            == 2
        )
    assert dispatch.await_count == 2


@pytest.mark.asyncio
async def test_threshold_change_preserves_manual_resolution(ctx, monkeypatch):
    client, factory = ctx
    model_id, _, _ = await _setup(client, monkeypatch, {"a": _normal(3, seed=4)})
    alert = (await _alerts(client, model_id))["items"][0]
    response = await client.patch(f"/alerts/{alert['id']}/resolve")
    assert response.status_code == 200
    async with factory() as session:
        original = (await session.get(Alert, alert["id"])).resolved_at
    monkeypatch.setattr(drift_engine.settings, "psi_yellow_threshold", 100.0)
    monkeypatch.setattr(drift_engine.settings, "psi_red_threshold", 100.0)
    await _analyze(client, model_id)
    async with factory() as session:
        row = await session.get(Alert, alert["id"])
        assert row.superseded_at is not None
        assert row.resolved_at == original


@pytest.mark.asyncio
async def test_threshold_change_only_clears_non_red_feature(ctx, monkeypatch):
    client, _ = ctx
    model_id, scores, dispatch = await _setup(
        client, monkeypatch, {"a": _normal(1.5, seed=4), "b": _normal(5, seed=5)}
    )
    assert all(score["severity"] == "red" for score in scores.values())
    assert scores["a"]["psi"] < scores["b"]["psi"]
    before = {
        row["feature_name"]: row["id"]
        for row in (await _alerts(client, model_id))["items"]
    }
    monkeypatch.setattr(
        drift_engine.settings,
        "psi_red_threshold",
        (scores["a"]["psi"] + scores["b"]["psi"]) / 2,
    )
    scores = await _analyze(client, model_id)
    assert scores["a"]["severity"] != "red"
    assert scores["b"]["severity"] == "red"
    assert [row["id"] for row in (await _alerts(client, model_id))["items"]] == [
        before["b"]
    ]
    assert [row["id"] for row in (await _alerts(client, model_id, True))["items"]] == [
        before["a"]
    ]
    assert dispatch.await_count == 2


@pytest.mark.asyncio
async def test_reanalysis_still_red_keeps_same_alert(ctx, monkeypatch):
    client, factory = ctx
    model_id, _, dispatch = await _setup(client, monkeypatch, {"a": _normal(3, seed=4)})
    before = await _alerts(client, model_id)
    assert before["total"] == 1
    assert (await _analyze(client, model_id))["a"]["severity"] == "red"
    assert await _alerts(client, model_id) == before
    async with factory() as session:
        assert (
            len(
                (
                    await session.scalars(
                        select(Alert).where(Alert.model_id == model_id)
                    )
                ).all()
            )
            == 1
        )
    assert dispatch.await_count == 1
