from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.alert import Alert

logger = logging.getLogger(__name__)


async def create_alert(
	db: AsyncSession,
	model_id: UUID,
	feature_name: str | None,
	alert_type: str,
	severity: str,
	message: str,
) -> Alert:
	"""Create an alert row and dispatch webhook asynchronously.

	Args:
		db: Database session.
		model_id: Monitored model identifier.
		feature_name: Feature associated with the alert.
		alert_type: Alert type value.
		severity: Alert severity.
		message: Human-readable message.

	Returns:
		Persisted alert ORM object.
	"""
	alert = Alert(
		model_id=model_id,
		feature_name=feature_name,
		alert_type=alert_type,
		severity=severity,
		message=message,
	)
	db.add(alert)
	await db.commit()
	await db.refresh(alert)

	asyncio.create_task(_dispatch_alert(alert))
	return alert


async def _dispatch_alert(alert: Alert) -> None:
	"""Dispatch alert payload to configured webhook URL.

	Args:
		alert: Persisted alert object.
	"""
	webhook_url = settings.alert_webhook_url.strip()
	if not webhook_url:
		logger.warning("ALERT_WEBHOOK_URL not configured; skipping dispatch for alert %s", alert.id)
		return

	payload = {
		"alert_id": str(alert.id),
		"model_id": str(alert.model_id),
		"feature_name": alert.feature_name,
		"alert_type": alert.alert_type,
		"severity": alert.severity,
		"message": alert.message,
		"created_at": alert.created_at.isoformat(),
	}

	try:
		async with httpx.AsyncClient(timeout=5.0) as client:
			response = await client.post(webhook_url, json=payload)
			if 200 <= response.status_code < 300:
				logger.info("Alert webhook dispatched successfully for alert %s", alert.id)
			else:
				logger.error(
					"Alert webhook failed for alert %s with status %s",
					alert.id,
					response.status_code,
				)
	except Exception:  # noqa: BLE001
		logger.exception("Alert webhook dispatch error for alert %s", alert.id)


async def list_alerts(
	db: AsyncSession,
	model_id: UUID | None = None,
	resolved: bool = False,
	page: int = 1,
	page_size: int = 50,
) -> list[Alert]:
	"""List alerts with optional model and resolved-state filtering.

	Args:
		db: Database session.
		model_id: Optional model filter.
		resolved: Whether to list resolved alerts.
		page: 1-based page number.
		page_size: Number of rows per page.

	Returns:
		List of alert rows.
	"""
	offset = (page - 1) * page_size
	query = select(Alert)

	if model_id is not None:
		query = query.where(Alert.model_id == model_id)

	if resolved:
		query = query.where(Alert.resolved_at.is_not(None))
	else:
		query = query.where(Alert.resolved_at.is_(None))

	query = query.order_by(Alert.created_at.desc()).offset(offset).limit(page_size)

	result = await db.execute(query)
	return list(result.scalars().all())


async def resolve_alert(db: AsyncSession, alert_id: UUID) -> Alert:
	"""Resolve an alert by setting resolved_at.

	Args:
		db: Database session.
		alert_id: Alert identifier.

	Returns:
		Updated alert row.

	Raises:
		HTTPException: If alert is not found.
	"""
	result = await db.execute(select(Alert).where(Alert.id == alert_id))
	alert = result.scalar_one_or_none()
	if alert is None:
		raise HTTPException(status_code=404, detail="Alert not found")

	alert.resolved_at = datetime.now(timezone.utc)
	await db.commit()
	await db.refresh(alert)
	return alert
