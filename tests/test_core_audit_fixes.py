import asyncio
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from app.config import Settings
from app.schemas.ingest import SnapshotPayload
from app.services.report_generator import _render_markdown_report
from app.services.snapshot_service import ingest_snapshot
from tests.test_reliability_fixes import ctx, _model, _reference  # noqa: F401


def test_psi_threshold_settings() -> None:
    assert Settings(_env_file=None, DATABASE_URL="x").psi_red_threshold == 0.2
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DATABASE_URL="x", PSI_RED_THRESHOLD=0)
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            DATABASE_URL="x",
            PSI_YELLOW_THRESHOLD=0.3,
            PSI_RED_THRESHOLD=0.2,
        )


@pytest.mark.asyncio
async def test_retry_analysis_invalidates_report(ctx) -> None:  # noqa: F811
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    async with session_factory() as session:
        await ingest_snapshot(
            session,
            SnapshotPayload(
                model_id=UUID(model_id),
                timestamp=datetime(2026, 2, 2, 12, tzinfo=timezone.utc),
                features={"a": [10.0] * 200},
            ),
        )
    report = await client.post(
        f"/reports/{model_id}/generate", json={"week_start": "2026-02-02"}
    )
    assert report.status_code == 200, report.text
    with patch("app.services.alert_service._dispatch_alert", new=AsyncMock()):
        analysis = await client.post(f"/drift/{model_id}/2026-02-02/analyze")
    assert analysis.status_code == 200, analysis.text
    assert analysis.json()[0]["severity"] == "red"
    assert (await client.get(f"/reports/{model_id}/2026-02-02")).status_code == 404


@pytest.mark.asyncio
async def test_webhooks_run_in_parallel_without_an_open_transaction(
    ctx,  # noqa: F811
) -> None:
    client, session_factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    async with session_factory() as session:
        await ingest_snapshot(
            session,
            SnapshotPayload(
                model_id=UUID(model_id),
                timestamp=datetime(2026, 2, 2, 12, tzinfo=timezone.utc),
                features={"a": [10.0] * 200, "b": [10.0] * 200},
            ),
        )
    started: list[str] = []
    idle_in_transaction: list[int] = []
    both_started = asyncio.Event()

    async def dispatch(alert) -> None:
        started.append(alert.feature_name)
        if len(started) == 2:
            async with session_factory() as session:
                idle_in_transaction.append(
                    await session.scalar(
                        text(
                            "SELECT count(*) FROM pg_stat_activity WHERE "
                            "datname = current_database() "
                            "AND state = 'idle in transaction'"
                        )
                    )
                )
            both_started.set()
        # Sequential sending never reaches the second alert, so this times out.
        await asyncio.wait_for(both_started.wait(), timeout=5)

    with patch("app.services.alert_service._dispatch_alert", new=dispatch):
        response = await client.post(f"/drift/{model_id}/2026-02-02/analyze")

    assert response.status_code == 200, response.text
    assert sorted(started) == ["a", "b"]
    assert idle_in_transaction == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["a\nb", "a\tb", "a\x7fb"])
async def test_control_characters_in_feature_names_are_rejected(
    ctx,  # noqa: F811
    name,
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    response = await client.post(
        "/ingest/reference",
        json={"model_id": model_id, "features": {name: [1.0, 2.0]}},
    )
    assert response.status_code == 422


def test_report_table_escapes_pipes_in_feature_names() -> None:
    markdown = _render_markdown_report(
        model_name="m",
        week_start=date(2026, 2, 2),
        week_end=date(2026, 2, 8),
        overall_score=None,
        ranked_rows=[{"feature_name": "a|b", "ks_statistic": 0.0}],
        output_drift=None,
        recommendations=[],
        red_count=0,
        yellow_count=0,
        green_count=0,
    )
    assert "| a\\|b | " in markdown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "week_start,status", [("9999-12-26", 422), ("9999-12-25", 200)]
)
async def test_week_start_cannot_overflow_the_calendar(
    ctx,  # noqa: F811
    week_start,
    status,
) -> None:
    client, _ = ctx
    model_id = await _model(client)
    response = await client.post(
        f"/reports/{model_id}/generate", json={"week_start": week_start}
    )
    assert response.status_code == status, response.text
