"""Regression coverage for readiness, database errors, and narration setup."""

from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

import app.database as database
from model_health_agent.narration import narrate_report
from tests.test_reliability_fixes import run_migrations  # noqa: F401


@pytest.mark.asyncio
async def test_readiness_requires_installed_head(monkeypatch):
    try:
        await database.check_database_readiness()
        scripts = Mock()
        scripts.from_config.return_value.get_heads.return_value = ["not-the-head"]
        monkeypatch.setattr(database, "ScriptDirectory", scripts)
        with pytest.raises(RuntimeError, match="migration head"):
            await database.check_database_readiness()
    finally:
        await database.engine.dispose()


@pytest.mark.asyncio
async def test_database_error_response(async_client, monkeypatch):
    monkeypatch.setattr(
        "app.routers.models.list_models", AsyncMock(side_effect=SQLAlchemyError("boom"))
    )
    response = await async_client.get("/models")
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_ERROR"
    assert "boom" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing",
    ["DRIFTWATCH_LLM_BASE_URL", "DRIFTWATCH_LLM_MODEL", "DRIFTWATCH_LLM_API_KEY"],
)
async def test_narration_requires_all_settings(missing, monkeypatch):
    for name in (
        "DRIFTWATCH_LLM_BASE_URL",
        "DRIFTWATCH_LLM_MODEL",
        "DRIFTWATCH_LLM_API_KEY",
    ):
        monkeypatch.setenv(name, "configured")
    monkeypatch.delenv(missing)
    client = Mock(post=AsyncMock())
    report = {"mode": "deterministic", "summary": "unchanged"}

    result = await narrate_report(report, client=client)

    client.post.assert_not_awaited()
    assert result is report
    assert result["mode"] == "deterministic"
