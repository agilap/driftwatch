"""Migration 0004 must round-trip without losing active alerts."""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

INSERT = text(
    "INSERT INTO alerts (model_id, window_date, feature_name, alert_type, severity, "
    "message, superseded_at) VALUES (CAST(:model AS uuid), '2026-01-01', 'a', "
    "'drift_red', 'red', :message, CAST(:superseded AS timestamptz))"
)


async def _alert_schema(connection):
    def read(sync_connection):
        inspector = inspect(sync_connection)
        return (
            {column["name"] for column in inspector.get_columns("alerts")},
            {c["name"] for c in inspector.get_unique_constraints("alerts")},
            {
                index["name"]: index
                for index in inspector.get_indexes("alerts")
                if index["unique"]
            },
        )

    return await connection.run_sync(read)


@pytest.mark.asyncio
async def test_migration_0004_up_down_up() -> None:
    url = os.environ["DATABASE_URL"]
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    engine = create_async_engine(url)
    model_id = str(uuid4())
    name = "uq_alert_model_feature_window_type"
    # Alembic uses asyncio.run, so run the synchronous migration in a thread.
    await asyncio.to_thread(command.upgrade, cfg, "head")
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO models (id, name) VALUES (CAST(:id AS uuid), :name)"),
                {"id": model_id, "name": f"migration-{model_id}"},
            )
            for message, superseded in (
                ("old", "2026-01-02T00:00:00Z"),
                ("older", "2026-01-01T00:00:00Z"),
                ("active", None),
            ):
                await connection.execute(
                    INSERT.bindparams(
                        model=model_id, message=message, superseded=superseded
                    )
                )
        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(
                    INSERT.bindparams(model=model_id, message="dup", superseded=None)
                )
        async with engine.connect() as connection:
            columns, constraints, indexes = await _alert_schema(connection)
        assert "superseded_at" in columns
        assert name not in constraints
        assert "superseded_at IS NULL" in str(
            indexes[name]["dialect_options"]["postgresql_where"]
        )

        await asyncio.to_thread(command.downgrade, cfg, "0003_statistical_correctness")
        async with engine.connect() as connection:
            columns, constraints, _ = await _alert_schema(connection)
            messages = (
                await connection.execute(
                    text(
                        "SELECT message FROM alerts WHERE model_id = CAST(:id AS uuid)"
                    ),
                    {"id": model_id},
                )
            ).all()
        assert "superseded_at" not in columns
        assert name in constraints
        assert [row.message for row in messages] == ["active"]

        await asyncio.to_thread(command.upgrade, cfg, "head")
        async with engine.connect() as connection:
            columns, constraints, indexes = await _alert_schema(connection)
        assert "superseded_at" in columns
        assert name not in constraints and name in indexes
        await asyncio.to_thread(command.check, cfg)
    finally:
        await asyncio.to_thread(command.upgrade, cfg, "head")
        async with engine.begin() as connection:
            await connection.execute(
                text("DELETE FROM models WHERE id = CAST(:id AS uuid)"),
                {"id": model_id},
            )
        await engine.dispose()
