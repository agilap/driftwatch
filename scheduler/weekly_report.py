from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.model_registry import ModelRegistry
from app.services.report_generator import generate_report

logger = logging.getLogger(__name__)

weekly_scheduler = AsyncIOScheduler(timezone="UTC")


async def run_weekly_reports_job() -> None:
    """Generate weekly reports for all models for the previous 7-day window."""
    today_utc = datetime.now(timezone.utc).date()
    week_end = today_utc - timedelta(days=1)
    week_start = week_end - timedelta(days=6)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ModelRegistry).order_by(ModelRegistry.name.asc())
        )
        models = list(result.scalars().all())

    for model in models:
        async with AsyncSessionLocal() as db:
            try:
                report = await generate_report(
                    db=db,
                    model_id=model.id,
                    week_start=week_start,
                    week_end=week_end,
                )
                logger.info(
                    "Weekly report generated model_id=%s score=%s",
                    model.id,
                    report.overall_score,
                )
            except Exception:
                await db.rollback()
                logger.exception("Weekly report failed model_id=%s", model.id)


def start_weekly_scheduler() -> None:
    """Start scheduler and register weekly report cron job."""
    if weekly_scheduler.running:
        return

    trigger = CronTrigger.from_crontab(settings.report_schedule_cron, timezone="UTC")
    weekly_scheduler.add_job(
        run_weekly_reports_job,
        trigger=trigger,
        id="weekly-health-report",
        replace_existing=True,
    )
    weekly_scheduler.start()


def stop_weekly_scheduler() -> None:
    """Stop scheduler on application shutdown."""
    if weekly_scheduler.running:
        weekly_scheduler.shutdown(wait=False)
