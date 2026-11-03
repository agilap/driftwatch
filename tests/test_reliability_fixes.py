"""Regression tests for stale-row replacement, alert supersede and input handling."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import AsyncGenerator
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.database import get_db
from app.main import app
from app.models.alert import Alert
from app.models.drift_score import DriftScore
from app.models.feature_importance import FeatureImportance
from app.models.reference_distribution import ReferenceDistribution
from app.models.snapshot import Snapshot
from app.services.importance_scorer import upsert_importances

DAY = "2026-02-02"
TIMESTAMP = f"{DAY}T12:00:00Z"
Context = tuple[AsyncClient, async_sessionmaker[AsyncSession]]


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def ctx() -> AsyncGenerator[Context, None]:
    engine = create_async_engine(os.environ["DATABASE_URL"], future=True)
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

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory
    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


def _normal(mean: float = 0.0, seed: int = 1, size: int = 1000) -> list[float]:
    return np.random.default_rng(seed).normal(mean, 1.0, size).tolist()


async def _model(client: AsyncClient) -> str:
    response = await client.post("/models", json={"name": f"m-{uuid4()}"})
    assert response.status_code == 201
    return response.json()["id"]


async def _reference(client: AsyncClient, model_id: str, **extra) -> dict:
    body = {
        "model_id": model_id,
        "features": {"a": _normal(seed=1), "b": _normal(seed=2)},
        **extra,
    }
    response = await client.post("/ingest/reference", json=body)
    assert response.status_code == 200, response.text
    return body


async def _snapshot(client: AsyncClient, model_id: str, features: dict, **extra):
    return await client.post(
        "/ingest/snapshot",
        json={
            "model_id": model_id,
            "timestamp": TIMESTAMP,
            "features": features,
            **extra,
        },
    )


async def _names(session_factory, table, model_id: str) -> set[str]:
    async with session_factory() as session:
        return set(
            await session.scalars(
                select(table.feature_name).where(table.model_id == model_id)
            )
        )


async def _alerts(client: AsyncClient, model_id: str, resolved: bool) -> list[dict]:
    response = await client.get(
        f"/alerts/{model_id}", params={"resolved": resolved, "page_size": 100}
    )
    assert response.status_code == 200
    return response.json()["items"]


# Alert paging


@pytest.mark.asyncio
async def test_alert_pages_are_complete_with_real_total(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    async with session_factory() as session:
        for index in range(120):
            session.add(
                Alert(
                    model_id=model_id,
                    window_date=date(2026, 1, 1),
                    feature_name=f"f{index}",
                    alert_type="drift_red",
                    severity="red",
                    message="m",
                    created_at=created_at,
                )
            )
        await session.commit()

    for path, params in (
        (f"/alerts/{model_id}", {}),
        ("/alerts", {"model_id": model_id}),
    ):
        seen: list[str] = []
        for page in range(1, 5):
            body = (
                await client.get(path, params={**params, "page": page, "page_size": 50})
            ).json()
            assert body["total"] == 120
            seen += [item["id"] for item in body["items"]]
        assert len(seen) == 120 and len(set(seen)) == 120
    resolved = await client.get(f"/alerts/{model_id}", params={"resolved": True})
    assert resolved.json()["total"] == 0


# Stale rows


@pytest.mark.asyncio
async def test_reference_reregistration_removes_absent_rows(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id, predictions=_normal(seed=3))
    assert await _names(session_factory, ReferenceDistribution, model_id) == {
        "a",
        "b",
        "__predictions__",
    }
    response = await client.post(
        "/ingest/reference", json={"model_id": model_id, "features": {"a": [1.0, 2.0]}}
    )
    assert response.status_code == 200
    assert await _names(session_factory, ReferenceDistribution, model_id) == {"a"}


@pytest.mark.asyncio
async def test_frozen_reference_deletes_nothing(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    body = await _reference(client, model_id)
    assert (await _snapshot(client, model_id, body["features"])).status_code == 200
    response = await client.post(
        "/ingest/reference",
        json={"model_id": model_id, "features": {"a": body["features"]["a"]}},
    )
    assert response.status_code == 200
    assert await _names(session_factory, ReferenceDistribution, model_id) == {"a", "b"}


@pytest.mark.asyncio
async def test_snapshot_repost_replaces_the_whole_window(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id, predictions=_normal(seed=3))
    first = await _snapshot(
        client,
        model_id,
        {"a": _normal(5), "b": _normal(seed=9)},
        predictions=_normal(seed=3),
    )
    assert first.status_code == 200
    assert await _alerts(client, model_id, resolved=False)

    # The kept feature is byte-identical, so removal alone must count as a change.
    second = await _snapshot(client, model_id, {"b": _normal(seed=9)})
    assert second.status_code == 200
    assert await _names(session_factory, Snapshot, model_id) == {"b"}
    assert await _names(session_factory, DriftScore, model_id) == {"b"}
    assert await _alerts(client, model_id, resolved=False) == []


@pytest.mark.asyncio
async def test_importance_replacement_deletes_missing_rows(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    body = await _reference(
        client,
        model_id,
        feature_importances={"a": 1.0, "b": 1.0},
        importance_method="manual",
    )
    async with session_factory() as session:
        session.add(
            FeatureImportance(
                model_id=model_id, feature_name="legacy", importance=0.5, method="shap"
            )
        )
        await session.commit()
    response = await client.post(
        f"/models/{model_id}/importances",
        json={"importances": {"a": 3.0, "b": 1.0}, "method": "shap"},
    )
    assert response.status_code == 200
    assert await _names(session_factory, FeatureImportance, model_id) == {"a", "b"}

    async with session_factory() as session:
        session.add(
            FeatureImportance(
                model_id=model_id, feature_name="legacy", importance=0.5, method="shap"
            )
        )
        await session.commit()
    response = await client.post(
        "/ingest/reference",
        json={
            **body,
            "feature_importances": {"a": 1.0, "b": 1.0},
            "importance_method": "manual",
        },
    )
    assert response.status_code == 200
    assert await _names(session_factory, FeatureImportance, model_id) == {"a", "b"}


@pytest.mark.asyncio
async def test_feature_set_change_without_importances_clears_them(
    ctx: Context, caplog
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    body = await _reference(
        client,
        model_id,
        feature_importances={"a": 1.0, "b": 1.0},
        importance_method="manual",
        predictions=_normal(seed=3),
    )
    # Dropping only predictions keeps the feature set, so importances remain valid.
    same_features = await client.post(
        "/ingest/reference",
        json={"model_id": model_id, "features": body["features"]},
    )
    assert same_features.status_code == 200
    assert await _names(session_factory, FeatureImportance, model_id) == {"a", "b"}

    with caplog.at_level(logging.WARNING, logger="app.services.reference_service"):
        changed = await client.post(
            "/ingest/reference",
            json={"model_id": model_id, "features": {"a": [1.0, 2.0]}},
        )
    assert changed.status_code == 200
    assert await _names(session_factory, FeatureImportance, model_id) == set()
    assert any("equal" in record.getMessage() for record in caplog.records)


# Importance coverage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "importances",
    [{"a": 1.0}, {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1.0, "__predictions__": 1.0}],
)
async def test_reference_rejects_non_matching_importances(
    ctx: Context, importances: dict
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": model_id,
            "features": {"a": [1.0, 2.0], "b": [1.0, 2.0]},
            "feature_importances": importances,
            "importance_method": "manual",
        },
    )
    assert response.status_code == 422
    assert await _names(session_factory, ReferenceDistribution, model_id) == set()


@pytest.mark.asyncio
async def test_frozen_reference_checks_importances_against_stored_features(
    ctx: Context,
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    body = await _reference(client, model_id, predictions=_normal(seed=3))
    await _snapshot(client, model_id, body["features"])
    subset = {"a": body["features"]["a"]}

    partial = await client.post(
        "/ingest/reference",
        json={
            "model_id": model_id,
            "features": subset,
            "feature_importances": {"a": 1.0},
            "importance_method": "manual",
        },
    )
    assert partial.status_code == 422
    assert "'b'" in partial.json()["detail"]

    complete = await client.post(
        "/ingest/reference",
        json={
            "model_id": model_id,
            "features": subset,
            "feature_importances": {"a": 2.0, "b": 0.0},
            "importance_method": "manual",
        },
    )
    assert complete.status_code == 200, complete.text
    assert await _names(session_factory, FeatureImportance, model_id) == {"a", "b"}


@pytest.mark.asyncio
async def test_importances_endpoint_requires_reference_and_coverage(
    ctx: Context,
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    url = f"/models/{model_id}/importances"
    payload = {"importances": {"a": 1.0, "b": 0.0}, "method": "shap"}

    assert (await client.post(url, json=payload)).status_code == 409

    await _reference(client, model_id, predictions=_normal(seed=3))
    bad = await client.post(
        url, json={"importances": {"a": 1.0, "ghost": 1.0}, "method": "manual"}
    )
    assert bad.status_code == 422
    assert "'b'" in bad.json()["detail"] and "'ghost'" in bad.json()["detail"]
    assert await _names(session_factory, FeatureImportance, model_id) == set()

    good = await client.post(url, json=payload)
    assert good.status_code == 200
    body = good.json()
    assert body["total"] == 2
    assert {item["feature_name"]: item["importance"] for item in body["items"]} == {
        "a": 1.0,
        "b": 0.0,
    }
    assert {item["method"] for item in body["items"]} == {"shap"}


@pytest.mark.asyncio
async def test_upsert_importances_replaces_and_returns_sorted(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    async with session_factory() as session:
        first = await upsert_importances(
            session, model_id, {"b": 0.75, "a": 0.25}, "shap"
        )
        assert [row.feature_name for row in first] == ["a", "b"]
    async with session_factory() as session:
        second = await upsert_importances(
            session, model_id, {"a": 0.5, "b": 0.5}, "manual"
        )
        assert [(r.importance, r.method) for r in second] == [
            (0.5, "manual"),
            (0.5, "manual"),
        ]


# Alert supersede


async def _red_window(client: AsyncClient, model_id: str) -> dict:
    await _reference(client, model_id)
    features = {"a": _normal(3, seed=4), "b": _normal(seed=5)}
    assert (await _snapshot(client, model_id, features)).status_code == 200
    return features


@pytest.mark.asyncio
async def test_corrected_data_leaves_no_active_alerts(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    with patch(
        "app.services.alert_service._dispatch_alert", new=AsyncMock()
    ) as dispatch:
        await _red_window(client, model_id)
        first = await _alerts(client, model_id, resolved=False)
        assert [alert["feature_name"] for alert in first] == ["a"]
        assert dispatch.await_count == 1
        corrected = {"a": _normal(seed=6), "b": _normal(seed=5)}
        assert (await _snapshot(client, model_id, corrected)).status_code == 200
    assert await _alerts(client, model_id, resolved=False) == []
    assert [a["id"] for a in await _alerts(client, model_id, resolved=True)] == [
        first[0]["id"]
    ]
    assert dispatch.await_count == 1
    async with session_factory() as session:
        row = await session.get(Alert, first[0]["id"])
        assert row.superseded_at is not None
        assert row.resolved_at == row.superseded_at


@pytest.mark.asyncio
async def test_worse_data_raises_a_new_alert_with_new_psi(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    with patch(
        "app.services.alert_service._dispatch_alert", new=AsyncMock()
    ) as dispatch:
        await _red_window(client, model_id)
        old = (await _alerts(client, model_id, resolved=False))[0]
        worse = {"a": _normal(6, seed=7), "b": _normal(seed=5)}
        assert (await _snapshot(client, model_id, worse)).status_code == 200

    active = await _alerts(client, model_id, resolved=False)
    assert len(active) == 1 and active[0]["id"] != old["id"]
    psi = next(
        row["psi"]
        for row in (await client.get(f"/drift/{model_id}/{DAY}")).json()
        if row["feature_name"] == "a"
    )
    assert f"PSI={psi:.3f}" in active[0]["message"]
    assert active[0]["message"] != old["message"]
    assert [a["id"] for a in await _alerts(client, model_id, resolved=True)] == [
        old["id"]
    ]
    assert dispatch.await_count == 2


@pytest.mark.asyncio
async def test_identical_repost_changes_no_alerts(ctx: Context) -> None:
    client, _ = ctx
    model_id = await _model(client)
    with patch(
        "app.services.alert_service._dispatch_alert", new=AsyncMock()
    ) as dispatch:
        features = await _red_window(client, model_id)
        before = await _alerts(client, model_id, resolved=False)
        assert (await _snapshot(client, model_id, features)).status_code == 200
    assert await _alerts(client, model_id, resolved=False) == before
    assert await _alerts(client, model_id, resolved=True) == []
    assert dispatch.await_count == 1


@pytest.mark.asyncio
async def test_manual_resolve_survives_reanalysis_but_not_new_data(
    ctx: Context,
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    with patch(
        "app.services.alert_service._dispatch_alert", new=AsyncMock()
    ) as dispatch:
        features = await _red_window(client, model_id)
        alert = (await _alerts(client, model_id, resolved=False))[0]
        resolved = (await client.patch(f"/alerts/{alert['id']}/resolve")).json()
        assert (
            await client.post(f"/drift/{model_id}/{DAY}/analyze")
        ).status_code == 200
        assert (await _snapshot(client, model_id, features)).status_code == 200
        assert await _alerts(client, model_id, resolved=False) == []
        assert dispatch.await_count == 1

        worse = {"a": _normal(6, seed=7), "b": _normal(seed=5)}
        assert (await _snapshot(client, model_id, worse)).status_code == 200
    assert len(await _alerts(client, model_id, resolved=False)) == 1
    assert dispatch.await_count == 2
    async with session_factory() as session:
        row = await session.get(Alert, alert["id"])
        assert row.superseded_at is not None
        # The operator's own resolution time is not overwritten by superseding.
        assert row.resolved_at.isoformat() == resolved["resolved_at"].replace(
            "Z", "+00:00"
        )
        assert row.resolved_at != row.superseded_at


# Validation errors


@pytest.mark.asyncio
async def test_validation_errors_list_location_without_echoing_input(
    ctx: Context,
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": model_id,
            "timestamp": "2026-02-02T12:00:00",
            "features": {"a": [1.0]},
        },
    )
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Validation error"
    assert body["code"] == "VALIDATION_ERROR"
    assert body["errors"]
    for error in body["errors"]:
        assert set(error) == {"loc", "msg", "type"}
    assert body["errors"][0]["loc"][:2] == ["body", "timestamp"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "features", ['{"\\ud800":[1.0]}', '{"\\ud800":[]}', '{"\\ud800":[true]}']
)
async def test_lone_surrogate_feature_names_return_422(
    ctx: Context, features: str
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    response = await client.post(
        "/ingest/reference",
        content='{"model_id":"%s","features":%s}' % (model_id, features),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_lone_surrogate_model_name_returns_422(ctx: Context) -> None:
    client, _ = ctx
    response = await client.post(
        "/models",
        content='{"name":"\\ud800"}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


# Numeric input


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        [-1e308, 1e308],
        [1e16, 1e16 + 2],
        [1e308, 1e308],
        [1e16, 1e16],
        [1e308, 1e308, -1e308],
    ],
)
async def test_unrepresentable_reference_samples_return_422(
    ctx: Context, values: list[float]
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": model_id,
            "features": {"ok": [1.0] * len(values), "x": values},
        },
    )
    assert response.status_code == 422
    assert "'x'" in response.json()["detail"]
    assert await _names(session_factory, ReferenceDistribution, model_id) == set()


@pytest.mark.asyncio
async def test_unrepresentable_snapshot_samples_change_nothing(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    good = {"a": _normal(seed=4, size=2), "b": _normal(seed=5, size=2)}
    assert (await _snapshot(client, model_id, good)).status_code == 200
    async with session_factory() as session:
        before = list(
            await session.scalars(
                select(Snapshot.stats).order_by(Snapshot.feature_name)
            )
        )
    response = await _snapshot(
        client, model_id, {"a": [0.5, 0.75], "b": [1e308, 1e308]}
    )
    assert response.status_code == 422
    assert "'b'" in response.json()["detail"]
    async with session_factory() as session:
        after = list(
            await session.scalars(
                select(Snapshot.stats).order_by(Snapshot.feature_name)
            )
        )
    assert after == before


# Other input handling


@pytest.mark.asyncio
async def test_snapshot_predictions_require_reference_predictions(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    body = await _reference(client, model_id)
    response = await _snapshot(
        client, model_id, body["features"], predictions=_normal(seed=3)
    )
    assert response.status_code == 422
    assert "prediction" in response.json()["detail"].lower()
    assert await _names(session_factory, Snapshot, model_id) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [{"name": "x\x00y"}, {"name": "ok", "version": "1\x002"}]
)
async def test_nul_in_model_fields_returns_422(ctx: Context, payload: dict) -> None:
    client, _ = ctx
    assert (await client.post("/models", json=payload)).status_code == 422


@pytest.mark.asyncio
async def test_nul_in_feature_names_returns_422(ctx: Context) -> None:
    client, _ = ctx
    model_id = await _model(client)
    bad = {"a\x00b": [1.0, 2.0]}
    assert (
        await client.post(
            "/ingest/reference", json={"model_id": model_id, "features": bad}
        )
    ).status_code == 422
    await _reference(client, model_id)
    assert (await _snapshot(client, model_id, bad)).status_code == 422
    assert (
        await client.post(
            f"/models/{model_id}/importances",
            json={"importances": {"a\x00b": 1.0}, "method": "manual"},
        )
    ).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("sample", [True, "1"])
async def test_samples_reject_bools_and_numeric_strings(
    ctx: Context, sample: object
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    for extra in (
        {"features": {"a": [sample, 2.0]}},
        {"features": {"a": [1.0, 2.0]}, "predictions": [sample, 2.0]},
    ):
        response = await client.post(
            "/ingest/reference", json={"model_id": model_id, **extra}
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_integer_samples_stay_accepted(ctx: Context) -> None:
    client, _ = ctx
    model_id = await _model(client)
    response = await client.post(
        "/ingest/reference",
        json={"model_id": model_id, "features": {"a": [1, 2.5]}, "predictions": [0, 1]},
    )
    assert response.status_code == 200


# Database outage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError(111, "Connect call failed"),
        socket.gaierror(-3, "Temporary failure in name resolution"),
        OSError("Multiple exceptions: connection refused"),
    ],
)
async def test_database_connection_failure_returns_503(error: OSError) -> None:
    async def _unavailable() -> AsyncGenerator[None, None]:
        raise error
        yield

    app.dependency_overrides[get_db] = _unavailable
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.get("/models")
    finally:
        app.dependency_overrides.pop(get_db, None)
    assert response.status_code == 503
    assert response.json() == {"detail": "Database error", "code": "DATABASE_ERROR"}


# Small fixes


@pytest.mark.asyncio
async def test_resolving_twice_keeps_first_resolved_at(ctx: Context) -> None:
    client, _ = ctx
    model_id = await _model(client)
    with patch("app.services.alert_service._dispatch_alert", new=AsyncMock()):
        await _red_window(client, model_id)
    alert = (await _alerts(client, model_id, resolved=False))[0]
    first = await client.patch(f"/alerts/{alert['id']}/resolve")
    second = await client.patch(f"/alerts/{alert['id']}/resolve")
    assert first.status_code == second.status_code == 200
    assert second.json()["resolved_at"] == first.json()["resolved_at"]


@pytest.mark.asyncio
async def test_alerts_for_unknown_model_return_404(ctx: Context) -> None:
    client, _ = ctx
    response = await client.get(f"/alerts/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


@pytest.mark.asyncio
async def test_frozen_reference_conflict_names_the_new_model_path(
    ctx: Context,
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    body = await _reference(client, model_id)
    await _snapshot(client, model_id, body["features"])
    response = await client.post(
        "/ingest/reference",
        json={"model_id": model_id, "features": {"a": _normal(4)}},
    )
    assert response.status_code == 409
    assert "new model under a new name" in response.json()["detail"]


def test_settings_have_no_secret_key() -> None:
    assert not hasattr(settings, "secret_key")


# Independent review fixes


@pytest.mark.asyncio
async def test_identical_repost_at_large_scale_changes_nothing(ctx: Context) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    rng = np.random.default_rng(11)
    features = {"a": rng.normal(3e23, 1e23, 500).tolist()}
    reference = {"model_id": model_id, "features": features}
    assert (await client.post("/ingest/reference", json=reference)).status_code == 200
    window = {"a": rng.normal(9e23, 1e23, 500).tolist()}

    with patch(
        "app.services.alert_service._dispatch_alert", new=AsyncMock()
    ) as dispatch:
        assert (await _snapshot(client, model_id, window)).status_code == 200
        active = await _alerts(client, model_id, resolved=False)
        assert len(active) == 1
        report = await client.post(
            f"/reports/{model_id}/generate", json={"week_start": DAY}
        )
        assert report.status_code == 200

        # Guards the premise: JSONB hands these floats back as Python ints.
        async with session_factory() as session:
            stored = await session.scalar(
                select(Snapshot.stats).where(Snapshot.model_id == model_id)
            )
        assert isinstance(stored["max"], int)

        assert (await _snapshot(client, model_id, window)).status_code == 200
        replay = await client.post("/ingest/reference", json=reference)

    assert replay.status_code == 200, replay.text
    assert await _alerts(client, model_id, resolved=False) == active
    assert await _alerts(client, model_id, resolved=True) == []
    assert dispatch.await_count == 1
    latest = await client.get(f"/reports/{model_id}/latest")
    assert latest.status_code == 200
    assert latest.json()["id"] == report.json()["id"]


@pytest.mark.asyncio
async def test_validation_never_logs_or_returns_sample_values(
    ctx: Context, caplog
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    with caplog.at_level(logging.WARNING, logger="driftwatch"):
        response = await client.post(
            "/ingest/reference",
            json={"model_id": model_id, "features": {"income": ["987654.125", 2.0]}},
        )
    assert response.status_code == 422
    assert "987654.125" not in response.text
    assert "987654.125" not in caplog.text
    # Feature names stay on purpose: they are schema names, not samples.
    assert response.json()["errors"][0]["loc"] == ["body", "features", "income", 0]
    assert "income" in caplog.text
    assert "float_type" in caplog.text


@pytest.mark.asyncio
async def test_red_window_of_only_zero_importance_features(ctx: Context) -> None:
    client, _ = ctx
    model_id = await _model(client)
    await _reference(
        client,
        model_id,
        feature_importances={"a": 1.0, "b": 0.0},
        importance_method="manual",
    )
    with patch("app.services.alert_service._dispatch_alert", new=AsyncMock()):
        response = await _snapshot(client, model_id, {"b": _normal(5, seed=4)})
    assert response.status_code == 200
    scores = (await client.get(f"/drift/{model_id}/{DAY}")).json()
    assert [(r["feature_name"], r["severity"]) for r in scores] == [("b", "red")]
    assert scores[0]["weighted_score"] == 0.0
    assert [a["feature_name"] for a in await _alerts(client, model_id, False)] == ["b"]
    report = await client.post(
        f"/reports/{model_id}/generate", json={"week_start": DAY}
    )
    assert report.json()["overall_score"] == pytest.approx(100.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("weight", [True, "1"])
async def test_importances_reject_bools_and_numeric_strings(
    ctx: Context, weight: object
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    body = await _reference(client, model_id)
    importances = {"a": weight, "b": 1.0}
    upload = await client.post(
        f"/models/{model_id}/importances",
        json={"importances": importances, "method": "manual"},
    )
    assert upload.status_code == 422
    reference = await client.post(
        "/ingest/reference",
        json={
            **body,
            "feature_importances": importances,
            "importance_method": "manual",
        },
    )
    assert reference.status_code == 422


@pytest.mark.asyncio
async def test_integer_importances_stay_accepted(ctx: Context) -> None:
    client, _ = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    response = await client.post(
        f"/models/{model_id}/importances",
        json={"importances": {"a": 3, "b": 1}, "method": "manual"},
    )
    assert response.status_code == 200
