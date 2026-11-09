from uuid import UUID

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from app.models.model_registry import ModelRegistry
from app.schemas.ingest import ReferencePayload
from app.services.importance_scorer import upsert_importances
from app.services.reference_service import register_reference
from tests.helpers import _model, _reference

pytestmark = pytest.mark.usefixtures("run_migrations")


@pytest.mark.asyncio
async def test_register_reference_locks_model(ctx):
    client, factory = ctx
    model_id = await _model(client)
    async with factory() as blocker, factory() as session:
        # NO KEY UPDATE allows foreign key checks while conflicting with FOR UPDATE.
        await blocker.execute(
            select(ModelRegistry)
            .where(ModelRegistry.id == model_id)
            .with_for_update(key_share=True)
        )
        await session.execute(text("SET LOCAL lock_timeout = '200ms'"))
        with pytest.raises(DBAPIError) as exc:
            await register_reference(
                session,
                ReferencePayload(model_id=UUID(model_id), features={"a": [1.0, 2.0]}),
            )
        assert exc.value.orig.sqlstate == "55P03"


@pytest.mark.asyncio
async def test_upsert_importances_locks_model(ctx):
    client, factory = ctx
    model_id = await _model(client)
    await _reference(client, model_id)
    async with factory() as blocker, factory() as session:
        await blocker.execute(
            select(ModelRegistry)
            .where(ModelRegistry.id == model_id)
            .with_for_update(key_share=True)
        )
        await session.execute(text("SET LOCAL lock_timeout = '200ms'"))
        with pytest.raises(DBAPIError) as exc:
            await upsert_importances(session, model_id, {"a": 0.5, "b": 0.5}, "manual")
        assert exc.value.orig.sqlstate == "55P03"
