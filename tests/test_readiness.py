from pathlib import Path
import subprocess
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from app import main


@pytest.mark.asyncio
async def test_readiness_success(async_client: AsyncClient, monkeypatch) -> None:
    check = AsyncMock()
    monkeypatch.setattr(main, "check_database_readiness", check)
    response = await async_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    check.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("schema mismatch"), OSError("offline")])
async def test_readiness_failure_preserves_liveness(
    async_client: AsyncClient, monkeypatch, error: Exception
) -> None:
    monkeypatch.setattr(main, "check_database_readiness", AsyncMock(side_effect=error))
    response = await async_client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert (await async_client.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_startup_rejects_unready_schema_and_disposes_engine(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "check_database_readiness",
        AsyncMock(side_effect=RuntimeError("unmigrated")),
    )
    dispose = AsyncMock()
    monkeypatch.setattr(main, "engine", SimpleNamespace(dispose=dispose))
    with pytest.raises(RuntimeError, match="unmigrated"):
        async with main.lifespan(main.app):
            pytest.fail("Application must not accept requests")
    dispose.assert_awaited_once()


def test_entrypoint_does_not_start_application_after_migration_failure(
    tmp_path,
) -> None:
    alembic = tmp_path / "alembic"
    alembic.write_text("#!/bin/sh\nexit 17\n")
    alembic.chmod(0o755)
    marker = tmp_path / "started"
    script = Path(__file__).resolve().parents[1] / "scripts/start.sh"
    result = subprocess.run(
        ["/bin/sh", str(script), "/usr/bin/touch", str(marker)],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 17
    assert not marker.exists()
