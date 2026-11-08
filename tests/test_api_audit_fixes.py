from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_alert_response_includes_window_date(async_client):
    model_id = uuid4()
    alert = SimpleNamespace(
        id=uuid4(),
        model_id=model_id,
        window_date=date(2026, 1, 7),
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift",
        resolved_at=None,
        created_at=datetime.now(timezone.utc),
    )
    with patch(
        "app.routers.alerts.list_alerts", new=AsyncMock(return_value=[alert])
    ), patch("app.routers.alerts.count_alerts", new=AsyncMock(return_value=1)):
        response = await async_client.get("/alerts")
    assert response.status_code == 200
    assert response.json()["items"][0]["window_date"] == "2026-01-07"


@pytest.mark.asyncio
async def test_latest_drift_excludes_predictions_from_counts(async_client):
    model_id = uuid4()
    rows = [
        SimpleNamespace(feature_name=name, psi=0.5, weighted_score=0.5, severity="red")
        for name in ("income", "__predictions__")
    ]
    with patch(
        "app.routers.drift.get_model", new=AsyncMock(return_value=object())
    ), patch(
        "app.routers.drift.get_latest_drift_window",
        new=AsyncMock(return_value=date(2026, 1, 7)),
    ), patch(
        "app.routers.drift.get_drift_scores_for_window",
        new=AsyncMock(return_value=rows),
    ):
        response = await async_client.get(f"/drift/{model_id}/latest")
    assert response.status_code == 200
    assert response.json()["red_count"] == 1
    assert len(response.json()["features"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "window,expected", [(date(2026, 1, 7), "2026-01-07"), (None, None)]
)
async def test_webhook_serializes_window_date(window, expected):
    from app.services.alert_service import _dispatch_alert

    alert = SimpleNamespace(
        id=uuid4(),
        model_id=uuid4(),
        window_date=window,
        feature_name="income",
        alert_type="drift_red",
        severity="red",
        message="Drift",
        created_at=datetime.now(timezone.utc),
    )
    with patch(
        "app.services.alert_service.settings.alert_webhook_url", "https://example.com"
    ), patch("httpx.AsyncClient.post", new=AsyncMock()) as post:
        await _dispatch_alert(alert)
    assert post.await_args.kwargs["json"]["window_date"] == expected
