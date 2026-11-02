"""Exercise persisted-count analysis and HTTP serialization without PostgreSQL."""

from datetime import date
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app
from app.services.drift_engine import run_drift_analysis


def _result(rows=None, scalar=None):
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar
    result.scalars.return_value.all.return_value = rows
    return result


def _analysis_session(mismatched=False, small_samples=False):
    model_id = uuid4()
    reference = {
        "bin_edges": [-2, 0, 2],
        "counts": [3e12, 1e12],
        "underflow": 0,
        "overflow": 0,
    }
    shifted = {**reference, "counts": [1e12, 3e12]}
    if mismatched:
        shifted["bin_edges"] = [10, 12, 14]
    if small_samples:
        shifted.update(counts=[20, 28], underflow=1, overflow=1)
    references = [
        SimpleNamespace(feature_name=name, distribution=reference)
        for name in ["a", "b"]
    ]
    snapshots = [
        SimpleNamespace(feature_name="a", distribution=shifted),
        SimpleNamespace(feature_name="b", distribution=reference),
    ]
    session = AsyncMock()
    session.add = MagicMock()
    session.execute.side_effect = [
        _result(scalar=SimpleNamespace(id=model_id)),
        _result(rows=references),
        _result(rows=snapshots),
        _result(),
        _result(),
    ]
    return model_id, session


@pytest.mark.asyncio
async def test_analysis_uses_counts_and_analyzed_importance_scope() -> None:
    model_id, session = _analysis_session()
    with patch(
        "app.services.importance_scorer.load_importances",
        new=AsyncMock(return_value={"a": 2, "unrelated": 98}),
    ), patch(
        "app.services.alert_service.create_alert",
        new=AsyncMock(return_value=SimpleNamespace(_dispatch_pending=False)),
    ):
        results = await run_drift_analysis(session, model_id, date(2026, 1, 1))
    assert len(results) == 2
    assert results[0].ks_statistic == pytest.approx(0.5)
    assert results[0].psi > 0.2
    assert results[0].weighted_score == pytest.approx(1.0)
    assert results[1].psi == 0
    assert results[1].weighted_score == 0
    assert all(row.ks_pvalue is None for row in results)
    assert all(call.args[0].ks_pvalue is None for call in session.add.call_args_list)
    session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_small_snapshot_warns_without_suppressing_alert(caplog) -> None:
    model_id, session = _analysis_session(small_samples=True)
    with patch("app.services.drift_engine.settings.min_sample_warning", 200), patch(
        "app.services.importance_scorer.load_importances",
        new=AsyncMock(return_value={}),
    ), patch(
        "app.services.alert_service.create_alert",
        new=AsyncMock(return_value=SimpleNamespace(_dispatch_pending=False)),
    ) as create_alert, caplog.at_level(
        logging.WARNING, logger="app.services.drift_engine"
    ):
        results = await run_drift_analysis(session, model_id, date(2026, 1, 1))

    assert len(caplog.records) == 1
    assert "Feature 'a' snapshot count 50" in caplog.text
    assert "PSI is inflated by sampling noise at this size" in caplog.text
    assert results[0].severity == "red"
    create_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_edge_mismatch_is_http_409_without_writes() -> None:
    model_id, session = _analysis_session(mismatched=True)

    async def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "app.services.importance_scorer.load_importances",
            new=AsyncMock(return_value={}),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(f"/drift/{model_id}/2026-01-01/analyze")
        assert response.status_code == 409
        assert "re-ingest" in response.json()["detail"]
        session.add.assert_not_called()
        session.commit.assert_not_awaited()
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_histogram_analysis_http_response_preserves_null_pvalue() -> None:
    model_id, session = _analysis_session()

    async def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        with patch(
            "app.services.importance_scorer.load_importances",
            new=AsyncMock(return_value={}),
        ), patch(
            "app.services.alert_service.create_alert",
            new=AsyncMock(return_value=SimpleNamespace(_dispatch_pending=False)),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                response = await client.post(f"/drift/{model_id}/2026-01-01/analyze")
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert all(row["ks_pvalue"] is None for row in response.json())
    finally:
        app.dependency_overrides.pop(get_db, None)
