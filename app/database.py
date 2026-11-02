from collections.abc import AsyncGenerator
from pathlib import Path

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for request-scoped dependencies."""
    async with AsyncSessionLocal() as session:
        yield session


async def check_database_readiness() -> None:
    """Require a reachable database with the installed migration heads applied."""
    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    expected = set(ScriptDirectory.from_config(config).get_heads())
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
        current = await connection.run_sync(
            lambda sync_connection: set(
                MigrationContext.configure(sync_connection).get_current_heads()
            )
        )
    if current != expected:
        raise RuntimeError("Database schema is not at the installed migration head")
