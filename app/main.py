import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.routers import alerts, drift, ingest, models, reports
from scheduler.weekly_report import start_weekly_scheduler, stop_weekly_scheduler

logger = logging.getLogger("driftwatch")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown lifecycle events."""
    logger.info("DriftWatch API starting up")
    start_weekly_scheduler()
    yield
    stop_weekly_scheduler()
    logger.info("DriftWatch API shutting down")


app = FastAPI(title="DriftWatch", version="0.1.0", lifespan=lifespan)

app.include_router(models.router, prefix="/models")
app.include_router(ingest.router, prefix="/ingest")
app.include_router(ingest.model_importance_router)
app.include_router(drift.router, prefix="/drift")
app.include_router(reports.router, prefix="/reports")
app.include_router(alerts.router, prefix="/alerts")


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness endpoint used by local and CI smoke checks."""
    return {"status": "ok", "version": "0.1.0"}
