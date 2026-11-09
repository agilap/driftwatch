import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


def _require_test_database() -> str:
    """Refuse to run against anything but a dedicated *_test database: tests TRUNCATE."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.exit(
            "TEST_DATABASE_URL is not set. Tests TRUNCATE tables, so they need a "
            "dedicated database whose name ends in _test."
        )
    try:
        name = make_url(url).database or ""
    except ArgumentError:
        pytest.exit("TEST_DATABASE_URL is not a valid database URL.")
    if not name.endswith("_test"):
        pytest.exit(
            f"TEST_DATABASE_URL points at database {name!r}; tests TRUNCATE tables, "
            "so its name must end in _test."
        )
    return url


# Must run before any app import: app.config reads DATABASE_URL at import time.
os.environ["DATABASE_URL"] = _require_test_database()
# Tests opt in to auth; a developer's own key must not turn every request into a 401.
os.environ["DRIFTWATCH_API_KEY"] = ""

from app.database import get_db  # noqa: E402
from app.main import app  # noqa: E402


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


@pytest_asyncio.fixture(autouse=True)
async def background_session_factory(monkeypatch):
    """Keep background workers on a test-owned engine and event loop."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.config import settings

    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.routers.ingest.AsyncSessionLocal", factory)
    try:
        yield
    finally:
        await engine.dispose()


async def truncate_tables(session):
    from sqlalchemy import text

    await session.execute(
        text(
            "TRUNCATE TABLE alerts, health_reports, drift_scores, snapshots, "
            "feature_importances, reference_distributions, models RESTART IDENTITY CASCADE"
        )
    )


@pytest.fixture(scope="session")
def run_migrations():
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def ctx(run_migrations):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(os.environ["DATABASE_URL"], future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        await truncate_tables(session)
        await session.commit()

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory
    app.dependency_overrides.pop(get_db, None)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(run_migrations):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(os.environ["DATABASE_URL"], future=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        await truncate_tables(session)
        await session.commit()
        yield session
    await engine.dispose()
