from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.database import get_db
from app.main import app


@pytest_asyncio.fixture
async def test_db_override() -> AsyncGenerator[None, None]:
    """Override the database dependency for tests that do not require a real DB."""

    async def _override_get_db() -> AsyncGenerator[None, None]:
        yield None

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest_asyncio.fixture
async def async_client(test_db_override: None) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client bound to the FastAPI ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
