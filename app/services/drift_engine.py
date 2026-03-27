"""Drift engine service placeholder."""

from __future__ import annotations

import logging
from datetime import date
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def run_drift_analysis(db: AsyncSession, model_id: UUID, window_date: date) -> None:
	"""Queueable drift analysis stub for a given model and snapshot window."""
	logger.info("Drift analysis triggered for %s %s", model_id, window_date)
