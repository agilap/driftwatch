import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.routers import alerts, drift, ingest, models, reports
from scheduler.weekly_report import start_weekly_scheduler, stop_weekly_scheduler

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
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


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log one line per request with method/path/status/duration."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(_: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Return a standard 503 response for database failures."""
    logger.exception("Database error: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "Database error", "code": "DATABASE_ERROR"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a standard 422 payload for validation failures."""
    logger.warning("Validation error: %s", exc)
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "code": "VALIDATION_ERROR"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Return a standard 500 payload for unexpected failures."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "INTERNAL_SERVER_ERROR"},
    )


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
