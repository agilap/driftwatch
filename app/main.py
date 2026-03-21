import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.routers import alerts, ingest, models, reports

logger = logging.getLogger("driftwatch")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown lifecycle events."""
    logger.info("DriftWatch API starting up")
    yield
    logger.info("DriftWatch API shutting down")


app = FastAPI(title="DriftWatch", version="0.1.0", lifespan=lifespan)

app.include_router(models.router, prefix="/models")
app.include_router(ingest.router, prefix="/ingest")
app.include_router(reports.router)
app.include_router(alerts.router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness endpoint used by local and CI smoke checks."""
    return {"status": "ok", "version": "0.1.0"}
