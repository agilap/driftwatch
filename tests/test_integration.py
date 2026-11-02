from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import date, datetime
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.models.alert import Alert
from app.models.drift_score import DriftScore
from app.models.feature_importance import FeatureImportance
from app.models.model_registry import ModelRegistry
from app.models.snapshot import Snapshot
from app.services.drift_engine import run_drift_analysis

JANUARY_TS = "2026-01-15T12:00:00Z"
AUGUST_TS = "2026-08-15T12:00:00Z"
AUGUST_WINDOW_DATE = date(2026, 8, 15)
AUGUST_WEEK_START = date(2026, 8, 10)

JAN_REFERENCE_FEATURES = {
    "income": [51000, 52000, 53000, 51500, 52500, 53500, 50500, 52000, 54000, 50000],
    "credit_score": [705, 710, 715, 700, 720, 710, 708, 712, 706, 714],
    "loan_term_months": [36, 36, 36, 36, 36, 36, 36, 36, 36, 36],
}

JAN_SNAPSHOT_FEATURES = {
    "income": [51000, 52000, 53000, 51500, 52500, 53500, 50500, 52000, 54000, 50000],
    "credit_score": [705, 710, 715, 700, 720, 710, 708, 712, 706, 714],
    "loan_term_months": [36, 36, 36, 36, 36, 36, 36, 36, 36, 36],
}

AUG_SNAPSHOT_FEATURES = {
    "income": [36000, 37000, 38000, 39000, 40000, 37500, 36500, 38500, 39500, 35500],
    "credit_score": [635, 640, 645, 650, 655, 648, 642, 638, 652, 646],
    "loan_term_months": [36, 36, 36, 36, 36, 36, 36, 36, 36, 36],
}

FEATURE_IMPORTANCES = {
    "income": 0.45,
    "credit_score": 0.40,
    "loan_term_months": 0.15,
}


def _database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://driftwatch:driftwatch@localhost:5432/driftwatch",
    )


@pytest.fixture(scope="session", autouse=True)
def run_migrations() -> None:
    """Apply migrations for full integration tests."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _database_url())
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def integration_context() -> (
    AsyncGenerator[tuple[AsyncClient, async_sessionmaker[AsyncSession]], None]
):
    """Provide isolated HTTP client and DB session factory for each integration test."""
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

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory

    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


async def _create_model(client: AsyncClient, name: str) -> UUID:
    response = await client.post("/models", json={"name": name, "version": "v3"})
    assert response.status_code == 201
    return UUID(response.json()["id"])


async def _register_reference(client: AsyncClient, model_id: UUID) -> None:
    response = await client.post(
        "/ingest/reference",
        json={
            "model_id": str(model_id),
            "features": JAN_REFERENCE_FEATURES,
            "feature_importances": FEATURE_IMPORTANCES,
            "importance_method": "manual",
        },
    )
    assert response.status_code == 200


async def _ingest_snapshot(
    client: AsyncClient,
    model_id: UUID,
    timestamp: str,
    features: dict[str, list[float]],
) -> None:
    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": timestamp,
            "features": features,
        },
    )
    assert response.status_code == 200


async def _prepare_august_pipeline(client: AsyncClient, model_name: str) -> UUID:
    model_id = await _create_model(client, name=model_name)
    await _register_reference(client, model_id=model_id)
    await _ingest_snapshot(
        client, model_id=model_id, timestamp=JANUARY_TS, features=JAN_SNAPSHOT_FEATURES
    )
    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=date(2026, 1, 15)
    )
    await _ingest_snapshot(
        client, model_id=model_id, timestamp=AUGUST_TS, features=AUG_SNAPSHOT_FEATURES
    )
    return model_id


async def run_drift_analysis_for_date(
    client: AsyncClient, model_id: UUID, target_date: date
) -> None:
    """Run drift analysis synchronously through a direct DB session."""
    override = app.dependency_overrides[get_db]
    async_gen = override()
    db = await async_gen.__anext__()
    try:
        await run_drift_analysis(db=db, model_id=model_id, window_date=target_date)
    finally:
        await async_gen.aclose()


@pytest.mark.asyncio
async def test_full_pipeline_register_model(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_name = f"loan-scorer-v3-{uuid4()}"

    response = await client.post("/models", json={"name": model_name, "version": "v3"})

    assert response.status_code == 201
    body = response.json()
    model_id = UUID(body["id"])
    assert body["name"] == model_name

    async with session_factory() as session:
        row = await session.execute(
            select(ModelRegistry).where(ModelRegistry.id == model_id)
        )
        assert row.scalar_one_or_none() is not None


@pytest.mark.asyncio
async def test_full_pipeline_register_reference(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")

    await _register_reference(client, model_id)

    async with session_factory() as session:
        result = await session.execute(
            select(FeatureImportance).where(FeatureImportance.model_id == model_id)
        )
        rows = result.scalars().all()

    assert len(rows) == 3
    assert sum(row.importance for row in rows) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_full_pipeline_ingest_january_snapshot(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")
    await _register_reference(client, model_id)

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": JANUARY_TS,
            "features": JAN_SNAPSHOT_FEATURES,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["drift_triggered"] is True
    assert set(body["features_recorded"]) == {
        "income",
        "credit_score",
        "loan_term_months",
    }


@pytest.mark.asyncio
async def test_full_pipeline_drift_january_no_alert(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")
    await _register_reference(client, model_id)
    await _ingest_snapshot(client, model_id, JANUARY_TS, JAN_SNAPSHOT_FEATURES)

    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=date(2026, 1, 15)
    )

    async with session_factory() as session:
        result = await session.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == date(2026, 1, 15),
            )
        )
        drift_rows = result.scalars().all()

        alert_rows = await session.execute(
            select(Alert).where(Alert.model_id == model_id)
        )
        alerts = alert_rows.scalars().all()

    assert len(drift_rows) == 3
    assert all(row.severity == "green" for row in drift_rows)
    assert all(float(row.psi or 0.0) < 0.10 for row in drift_rows)
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_full_pipeline_ingest_august_snapshot(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _create_model(client, name=f"loan-scorer-v3-{uuid4()}")
    await _register_reference(client, model_id)

    response = await client.post(
        "/ingest/snapshot",
        json={
            "model_id": str(model_id),
            "timestamp": AUGUST_TS,
            "features": AUG_SNAPSHOT_FEATURES,
        },
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_full_pipeline_drift_august_detects_shift(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )

    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    async with session_factory() as session:
        result = await session.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == AUGUST_WINDOW_DATE,
            )
        )
        rows = {row.feature_name: row for row in result.scalars().all()}

        alerts_result = await session.execute(
            select(Alert).where(Alert.model_id == model_id)
        )
        alerts = alerts_result.scalars().all()

    assert rows["income"].severity == "red"
    assert float(rows["income"].psi or 0.0) > 0.20
    assert rows["credit_score"].severity in {"red", "yellow"}
    assert rows["loan_term_months"].severity in {"green", "yellow"}
    assert any(alert.alert_type == "drift_red" for alert in alerts)
    assert float(rows["income"].weighted_score or 0.0) > float(
        rows["loan_term_months"].weighted_score or 0.0
    )


@pytest.mark.asyncio
async def test_full_pipeline_alerts_created_correctly(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )
    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    response = await client.get(f"/alerts?model_id={model_id}")

    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert any(item["feature_name"] == "income" for item in items)
    assert any("PSI=" in item["message"] for item in items)


@pytest.mark.asyncio
async def test_full_pipeline_resolve_alert(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )
    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    list_response = await client.get(f"/alerts?model_id={model_id}")
    alert_id = list_response.json()["items"][0]["id"]

    resolve_response = await client.patch(f"/alerts/{alert_id}/resolve")
    assert resolve_response.status_code == 200
    assert resolve_response.json()["resolved_at"] is not None

    active_response = await client.get(f"/alerts?model_id={model_id}")
    active_ids = [item["id"] for item in active_response.json()["items"]]
    assert alert_id not in active_ids


@pytest.mark.asyncio
async def test_full_pipeline_generate_report(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )
    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    response = await client.post(
        f"/reports/{model_id}/generate",
        json={"week_start": AUGUST_WEEK_START.isoformat()},
    )

    assert response.status_code == 200
    body = response.json()
    assert float(body["overall_score"] or 0.0) < 80.0
    report_json = body["report_json"]
    assert "executive_summary" in report_json
    assert "feature_drift_summary" in report_json
    assert "output_distribution" in report_json
    assert "top_drifted_features" in report_json
    assert "feature_to_output_correlation" not in report_json
    assert "recommendations" in report_json
    assert report_json["top_drifted_features"][0]["feature_name"] == "income"
    assert any(
        "deciding to retrain" in recommendation.lower()
        or "income" in recommendation.lower()
        for recommendation in report_json["recommendations"]
    )


@pytest.mark.asyncio
async def test_full_pipeline_report_markdown_complete(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )
    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )
    await client.post(
        f"/reports/{model_id}/generate",
        json={"week_start": AUGUST_WEEK_START.isoformat()},
    )

    response = await client.get(f"/reports/{model_id}/latest")

    assert response.status_code == 200
    markdown = response.json()["report_markdown"]
    assert isinstance(markdown, str)
    assert markdown.strip()
    assert "## Feature Drift Summary" in markdown
    assert "income" in markdown


@pytest.mark.asyncio
async def test_get_drift_scores_endpoint(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, _ = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )
    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    response = await client.get(f"/drift/{model_id}/{AUGUST_WINDOW_DATE.isoformat()}")

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 3
    assert all(item["ks_pvalue"] is None for item in items)
    assert all(item["severity"] in {"green", "yellow", "red"} for item in items)


@pytest.mark.asyncio
async def test_idempotency_reingest_same_snapshot(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    client, session_factory = integration_context
    model_id = await _prepare_august_pipeline(
        client, model_name=f"loan-scorer-v3-{uuid4()}"
    )

    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    # Reingest the same August snapshot date.
    await _ingest_snapshot(
        client, model_id=model_id, timestamp=AUGUST_TS, features=AUG_SNAPSHOT_FEATURES
    )

    async with session_factory() as session:
        snap_result = await session.execute(
            select(Snapshot).where(
                Snapshot.model_id == model_id,
                Snapshot.window_date == AUGUST_WINDOW_DATE,
            )
        )
        snapshot_rows = snap_result.scalars().all()

    assert len(snapshot_rows) == 3

    await run_drift_analysis_for_date(
        client, model_id=model_id, target_date=AUGUST_WINDOW_DATE
    )

    async with session_factory() as session:
        drift_result = await session.execute(
            select(DriftScore).where(
                DriftScore.model_id == model_id,
                DriftScore.window_date == AUGUST_WINDOW_DATE,
            )
        )
        drift_rows = drift_result.scalars().all()

    assert len(drift_rows) == 3


@pytest.mark.asyncio
async def test_legacy_snapshot_edges_return_conflict_until_reingested(
    integration_context: tuple[AsyncClient, async_sessionmaker[AsyncSession]],
) -> None:
    from app.services.model_service import _compute_histogram

    client, session_factory = integration_context
    model_id = await _create_model(client, name=f"legacy-edges-{uuid4()}")
    await _register_reference(client, model_id)
    await _ingest_snapshot(client, model_id, AUGUST_TS, AUG_SNAPSHOT_FEATURES)
    async with session_factory() as session:
        row = (
            await session.execute(
                select(Snapshot).where(
                    Snapshot.model_id == model_id,
                    Snapshot.feature_name == "income",
                )
            )
        ).scalar_one()
        row.distribution = _compute_histogram(AUG_SNAPSHOT_FEATURES["income"])
        await session.commit()

    response = await client.post(f"/drift/{model_id}/{AUGUST_WINDOW_DATE}/analyze")
    assert response.status_code == 409
    assert "re-ingest" in response.json()["detail"]

    await _ingest_snapshot(client, model_id, AUGUST_TS, AUG_SNAPSHOT_FEATURES)
    response = await client.post(f"/drift/{model_id}/{AUGUST_WINDOW_DATE}/analyze")
    assert response.status_code == 200
    assert len(response.json()) == 3
    assert all(row["ks_pvalue"] is None for row in response.json())


async def _demo_lifecycle(client, session_factory, tmp_path, monkeypatch):
    from app.routers import ingest
    from examples.loan_risk_demo.config import (
        BATCH_SIZE,
        FEATURES,
        REFERENCE_SEED,
        REFERENCE_SIZE,
        SCENARIOS,
    )
    from examples.loan_risk_demo.generate_data import feature_payload, generate_data
    from examples.loan_risk_demo.train_model import train_model

    monkeypatch.setattr(ingest, "AsyncSessionLocal", session_factory)
    reference, labels = generate_data(REFERENCE_SIZE, REFERENCE_SEED)
    model, _ = train_model(reference, labels, tmp_path / "model.joblib")
    coefficients = abs(model.named_steps["logisticregression"].coef_[0])
    model_id = await _create_model(client, f"demo-{uuid4()}")
    reference_payload = {
        "model_id": str(model_id),
        "features": feature_payload(reference),
        "predictions": model.predict_proba(reference)[:, 1].tolist(),
        "feature_importances": dict(zip(FEATURES, coefficients.tolist())),
        "importance_method": "coefficient",
    }
    assert (
        await client.post("/ingest/reference", json=reference_payload)
    ).status_code == 200
    snapshots = {}
    for scenario, (seed, shifted, window) in SCENARIOS.items():
        batch, _ = generate_data(BATCH_SIZE, seed, shifted)
        snapshots[scenario] = {
            "model_id": str(model_id),
            "timestamp": f"{window}T12:00:00Z",
            "features": feature_payload(batch),
            "predictions": model.predict_proba(batch)[:, 1].tolist(),
        }
        assert (
            await client.post("/ingest/snapshot", json=snapshots[scenario])
        ).status_code == 200
        assert (
            await client.post(
                f"/reports/{model_id}/generate", json={"week_start": window}
            )
        ).status_code == 200
    return model_id, reference_payload, snapshots, coefficients


@pytest.mark.asyncio
async def test_demo_lifecycle_statistics_and_state(
    integration_context, tmp_path, monkeypatch
):
    import numpy as np

    from app.config import settings
    from app.models.reference_distribution import ReferenceDistribution
    from app.services.drift_engine import (
        classify_severity,
        compute_binned_ks,
        compute_js_divergence_from_counts,
        compute_psi_from_counts,
    )
    from examples.loan_risk_demo.config import BATCH_SIZE, FEATURES, SCENARIOS

    client, session_factory = integration_context
    model_id, _, _, coefficients = await _demo_lifecycle(
        client, session_factory, tmp_path, monkeypatch
    )
    async with session_factory() as session:
        references = {
            r.feature_name: r
            for r in (await session.scalars(select(ReferenceDistribution))).all()
        }
        snapshots = {
            (r.window_date, r.feature_name): r
            for r in (await session.scalars(select(Snapshot))).all()
        }
        scores = {
            (r.window_date, r.feature_name): r
            for r in (await session.scalars(select(DriftScore))).all()
        }
        importances = {
            r.feature_name: r
            for r in (await session.scalars(select(FeatureImportance))).all()
        }
        alerts = (await session.scalars(select(Alert))).all()

    assert set(importances) == set(FEATURES)
    for name, coefficient in zip(FEATURES, coefficients):
        assert importances[name].method == "coefficient"
        assert importances[name].importance == pytest.approx(
            coefficient / sum(coefficients)
        )
    assert len(scores) == len(snapshots) == 12
    reports = {}
    for scenario, (_, _, window) in SCENARIOS.items():
        day = date.fromisoformat(window)
        response = await client.get(f"/drift/{model_id}/{window}")
        assert response.status_code == 200
        api = {row["feature_name"]: row for row in response.json()}
        assert set(api) == set(FEATURES) | {"__predictions__"}
        for name, item in api.items():
            row = scores[day, name]
            assert item["id"] == str(row.id)
            assert item["model_id"] == str(row.model_id)
            assert item["window_date"] == row.window_date.isoformat()
            assert (
                datetime.fromisoformat(item["computed_at"].replace("Z", "+00:00"))
                == row.computed_at
            )
            ref, snap = references[name].distribution, snapshots[day, name].distribution
            assert ref["bin_edges"] == snap["bin_edges"]
            counts = [
                np.asarray([h["underflow"], *h["counts"], h["overflow"]])
                for h in (ref, snap)
            ]
            assert sum(counts[1]) == snapshots[day, name].sample_count == BATCH_SIZE
            assert row.psi == pytest.approx(compute_psi_from_counts(*counts))
            assert row.js_divergence == pytest.approx(
                compute_js_divergence_from_counts(*counts)
            )
            assert row.ks_statistic == pytest.approx(compute_binned_ks(*counts))
            assert row.ks_pvalue is None
            assert row.severity == classify_severity(row.psi)
            expected = "green" if scenario == "A" or name in FEATURES[3:] else "red"
            assert row.severity == expected
            weight = (
                0
                if name == "__predictions__"
                else importances[name].importance
                / sum(r.importance for r in importances.values())
                * min(row.psi / settings.psi_red_threshold, 1)
            )
            assert row.weighted_score == pytest.approx(weight)
            for field in (
                "psi",
                "js_divergence",
                "ks_statistic",
                "ks_pvalue",
                "weighted_score",
                "severity",
            ):
                assert item[field] == getattr(row, field)
        response = await client.get(f"/reports/{model_id}/{window}")
        assert response.status_code == 200
        reports[scenario] = response.json()
        report = reports[scenario]
        summary = report["report_json"]["feature_drift_summary"]
        assert len(summary) == 5
        assert {r["feature_name"] for r in summary} == set(FEATURES)
        assert report["overall_score"] == pytest.approx(
            100 * (1 - sum(scores[day, name].weighted_score for name in FEATURES))
        )
        assert report["report_json"]["output_distribution"][
            "output_drift_psi"
        ] == pytest.approx(scores[day, "__predictions__"].psi)
        assert "KS Statistic" in report["report_markdown"]
        assert all(name in report["report_markdown"] for name in FEATURES)
    ranked = reports["B"]["report_json"]["feature_drift_summary"]
    assert [r["weighted_score"] for r in ranked] == sorted(
        (r["weighted_score"] for r in ranked), reverse=True
    )
    assert any(
        a["psi"] < b["psi"] for i, a in enumerate(ranked) for b in ranked[i + 1 :]
    ), "No pair has higher rank and lower PSI"
    assert reports["B"]["overall_score"] < reports["A"]["overall_score"]
    day_a, day_b = (date.fromisoformat(SCENARIOS[s][2]) for s in ("A", "B"))
    assert not [a for a in alerts if a.window_date == day_a]
    assert len(alerts) == 4
    assert {(a.feature_name, a.alert_type) for a in alerts} == {
        (name, "drift_red") for name in FEATURES[:3]
    } | {("__predictions__", "output_shift")}
    for alert in alerts:
        assert alert.window_date == day_b
        assert f"PSI={scores[day_b, alert.feature_name].psi:.3f}" in alert.message
        assert str(day_b) in alert.message


async def _demo_state(session_factory):
    from app.models.health_report import HealthReport
    from app.models.reference_distribution import ReferenceDistribution

    models = (
        ModelRegistry,
        ReferenceDistribution,
        FeatureImportance,
        Snapshot,
        DriftScore,
        HealthReport,
        Alert,
    )
    async with session_factory() as session:
        return {
            model.__tablename__: [
                {
                    column.name: getattr(row, column.name)
                    for column in model.__table__.columns
                }
                for row in (
                    await session.scalars(select(model).order_by(model.id))
                ).all()
            ]
            for model in models
        }


@pytest.mark.asyncio
async def test_demo_repeated_requests(integration_context, tmp_path, monkeypatch):
    from examples.loan_risk_demo.config import SCENARIOS

    client, session_factory = integration_context
    model_id, reference, snapshots, _ = await _demo_lifecycle(
        client, session_factory, tmp_path, monkeypatch
    )
    before = await _demo_state(session_factory)
    assert (await client.post("/ingest/reference", json=reference)).status_code == 200
    for scenario, (_, _, window) in SCENARIOS.items():
        assert (
            await client.post("/ingest/snapshot", json=snapshots[scenario])
        ).status_code == 200
        assert (
            await client.post(f"/drift/{model_id}/{window}/analyze")
        ).status_code == 200
        assert (
            await client.post(
                f"/reports/{model_id}/generate", json={"week_start": window}
            )
        ).status_code == 200
    after = await _demo_state(session_factory)
    assert {k: len(v) for k, v in before.items()} == {
        k: len(v) for k, v in after.items()
    }
    assert before["reference_distributions"] == after["reference_distributions"]
    assert before["snapshots"] == after["snapshots"]
    assert before["feature_importances"] == after["feature_importances"]
    for original, repeated in zip(before["health_reports"], after["health_reports"]):
        for field in ("overall_score", "report_json", "report_markdown"):
            assert original[field] == repeated[field]
    for original, repeated in zip(before["drift_scores"], after["drift_scores"]):
        for field in (
            "psi",
            "js_divergence",
            "ks_statistic",
            "ks_pvalue",
            "weighted_score",
            "severity",
        ):
            assert original[field] == repeated[field]
    alert_id = before["alerts"][0]["id"]
    assert (await client.patch(f"/alerts/{alert_id}/resolve")).status_code == 200
    assert (
        await client.post(f"/drift/{model_id}/{SCENARIOS['B'][2]}/analyze")
    ).status_code == 200
    final = await _demo_state(session_factory)
    assert len(final["alerts"]) == len(before["alerts"])
    assert (
        next(a for a in final["alerts"] if a["id"] == alert_id)["resolved_at"]
        is not None
    )


@pytest.mark.asyncio
async def test_demo_concurrent_requests(
    integration_context, tmp_path, monkeypatch, caplog
):
    import asyncio

    from examples.loan_risk_demo.config import SCENARIOS

    client, session_factory = integration_context
    model_id, reference, snapshots, _ = await _demo_lifecycle(
        client, session_factory, tmp_path, monkeypatch
    )
    before = await _demo_state(session_factory)
    window = SCENARIOS["B"][2]

    async def concurrent(url, payload=None):
        results = await asyncio.wait_for(
            asyncio.gather(
                *(client.post(url, json=payload) for _ in range(5)),
                return_exceptions=True,
            ),
            timeout=60,
        )
        assert all(
            not isinstance(result, BaseException) and result.status_code == 200
            for result in results
        ), results

    await concurrent("/ingest/snapshot", snapshots["B"])
    await concurrent(f"/drift/{model_id}/{window}/analyze")
    await concurrent(f"/reports/{model_id}/generate", {"week_start": window})
    fresh_id = await _create_model(client, f"demo-concurrent-{uuid4()}")
    assert (
        await client.post(
            "/ingest/reference", json={**reference, "model_id": str(fresh_id)}
        )
    ).status_code == 200
    await concurrent("/ingest/snapshot", {**snapshots["A"], "model_id": str(fresh_id)})
    fresh_window = date.fromisoformat(SCENARIOS["A"][2])
    async with session_factory() as session:
        for model in (Alert, DriftScore):
            await session.execute(
                delete(model).where(
                    model.model_id == fresh_id, model.window_date == fresh_window
                )
            )
        await session.commit()
    await concurrent(f"/drift/{fresh_id}/{fresh_window}/analyze")
    async with session_factory() as session:
        scores = (
            await session.scalars(
                select(DriftScore).where(
                    DriftScore.model_id == fresh_id,
                    DriftScore.window_date == fresh_window,
                )
            )
        ).all()
    assert sorted(score.feature_name for score in scores) == sorted(
        [*reference["features"], "__predictions__"]
    )
    await concurrent(
        f"/reports/{fresh_id}/generate", {"week_start": fresh_window.isoformat()}
    )
    after = await _demo_state(session_factory)
    assert (
        len(
            [
                report
                for report in after["health_reports"]
                if report["model_id"] == fresh_id
                and report["week_start"] == fresh_window
            ]
        )
        == 1
    )
    assert not any(
        "Drift analysis failed" in record.getMessage() for record in caplog.records
    )
    assert len([a for a in after["alerts"] if a["model_id"] == model_id]) == len(
        before["alerts"]
    )
    keys = {
        "reference_distributions": ("model_id", "feature_name"),
        "feature_importances": ("model_id", "feature_name"),
        "snapshots": ("model_id", "window_date", "feature_name"),
        "drift_scores": ("model_id", "window_date", "feature_name"),
        "health_reports": ("model_id", "week_start"),
        "alerts": ("model_id", "window_date", "feature_name", "alert_type"),
    }
    for table, fields in keys.items():
        natural_keys = [tuple(row[field] for field in fields) for row in after[table]]
        assert len(natural_keys) == len(set(natural_keys))


@pytest.mark.asyncio
async def test_mcp_tools_return_real_monitoring_results(
    integration_context, tmp_path, monkeypatch
):
    from mcp.shared.memory import create_connected_server_and_client_session

    from driftwatch_mcp.server import build_server
    from examples.loan_risk_demo.config import FEATURES, SCENARIOS

    client, session_factory = integration_context
    model_id, *_ = await _demo_lifecycle(client, session_factory, tmp_path, monkeypatch)
    shifted_day = date.fromisoformat(SCENARIOS["B"][2])
    async with session_factory() as session:
        stored = {
            r.feature_name: r
            for r in (
                await session.scalars(
                    select(DriftScore).where(DriftScore.window_date == shifted_day)
                )
            ).all()
        }
        alert_count = len((await session.scalars(select(Alert))).all())

    server = build_server(client)
    async with create_connected_server_and_client_session(
        server._mcp_server
    ) as mcp_session:

        async def call(tool, **arguments):
            result = await mcp_session.call_tool(tool, arguments)
            assert not result.isError, result.content
            return result.structuredContent

        models = await call("list_models")
        assert [m["id"] for m in models["items"]] == [str(model_id)]

        health = await call("get_model_health", model_id=str(model_id))
        report = await client.get(f"/reports/{model_id}/latest")
        assert health["status"] == "critical"
        assert health["latest_window"] == SCENARIOS["B"][2]
        assert health["severity_counts"] == {"red": 3, "yellow": 0, "green": 2}
        assert health["prediction_severity"] == "red"
        assert health["overall_score"] == report.json()["overall_score"]
        assert health["report_covers_latest_window"] is True

        drift = await call(
            "get_feature_drift", model_id=str(model_id), feature_name="annual_income"
        )
        assert drift["features"][0]["psi"] == stored["annual_income"].psi
        assert drift["prediction"]["psi"] == stored["__predictions__"].psi

        stable = await call(
            "get_feature_drift",
            model_id=str(model_id),
            window_date=SCENARIOS["A"][2],
        )
        assert {r["severity"] for r in stable["features"]} == {"green"}

        alerts = await call("get_active_alerts", model_id=str(model_id))
        assert alerts["total"] == alert_count == 4

        weekly = await call(
            "get_health_report",
            model_id=str(model_id),
            week_start=SCENARIOS["A"][2],
        )
        assert weekly["week_start"] == SCENARIOS["A"][2]
        assert weekly["overall_score"] > health["overall_score"]

        analysis = await call("analyze_model_health", model_id=str(model_id))
        ranked = [r["feature_name"] for r in analysis["problem_features"]]
        assert set(ranked) == set(FEATURES[:3])
        assert ranked == sorted(
            ranked, key=lambda n: stored[n].weighted_score, reverse=True
        )
        assert set(analysis["stable_features"]) == set(FEATURES[3:])
        assert len(analysis["active_alerts"]) == 4
