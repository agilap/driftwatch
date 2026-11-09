import logging
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.config import settings
from app.schemas.ingest import ReferencePayload
from app.schemas.model import ModelCreate


def test_httpx_logging_hides_webhook_urls():
    import app.main  # noqa: F401

    assert logging.getLogger("httpx").level == logging.WARNING


def test_name_length_boundaries():
    assert len(ModelCreate(name="x" * 255).name) == 255
    assert ReferencePayload(model_id=uuid4(), features={"x" * 255: [1.0]})
    with pytest.raises(ValidationError):
        ModelCreate(name="x" * 256)
    with pytest.raises(ValidationError):
        ReferencePayload(model_id=uuid4(), features={"x" * 256: [1.0]})


@pytest.mark.parametrize("name", [" a", "a "])
def test_feature_names_reject_surrounding_whitespace(name):
    with pytest.raises(ValidationError):
        ReferencePayload(model_id=uuid4(), features={name: [1.0]})


@pytest.mark.asyncio
async def test_request_log_uses_scope_path(async_client, caplog):
    with caplog.at_level(logging.INFO, logger="driftwatch"):
        response = await async_client.get(
            "/health", headers={"Host": "x/models/fake-admin-action?"}
        )
    assert response.status_code == 200
    assert "path=/health" in caplog.text
    assert "fake-admin-action" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["x" * 256, "m\n# Fake"])
async def test_invalid_model_name_returns_422(async_client, name):
    response = await async_client.post("/models", json={"name": name})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_long_feature_name_returns_422(async_client):
    response = await async_client.post(
        "/ingest/reference",
        json={"model_id": str(uuid4()), "features": {"x" * 256: [1.0]}},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/models", "/alerts", "/alerts/{model_id}"])
async def test_large_page_returns_422(async_client, monkeypatch, path):
    monkeypatch.setattr(
        "app.routers.dependencies.get_model", AsyncMock(return_value=object())
    )
    response = await async_client.get(
        path.format(model_id=uuid4()), params={"page": 1000001}
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/ingest/reference/{id}",
        "/ingest/snapshots/{id}",
        "/ingest/snapshots/{id}/2026-01-01",
        "/drift/{id}/latest",
        "/drift/{id}/2026-01-01",
        "/reports/{id}",
        "/reports/{id}/latest",
        "/reports/{id}/2026-01-01",
        "/alerts/{id}",
    ],
)
async def test_missing_model_returns_404(async_client, monkeypatch, path):
    monkeypatch.setattr(
        "app.routers.dependencies.get_model", AsyncMock(return_value=None)
    )
    response = await async_client.get(path.format(id=uuid4()))
    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


@pytest.mark.asyncio
async def test_auth_runs_before_missing_model_check(async_client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")
    get_model = AsyncMock(return_value=None)
    monkeypatch.setattr("app.routers.dependencies.get_model", get_model)
    response = await async_client.get(f"/alerts/{uuid4()}")
    assert response.status_code == 401
    get_model.assert_not_awaited()
