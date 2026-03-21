from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Text, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ModelRegistry(Base):
    """Monitored model registry table."""

    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )

    reference_distributions = relationship(
        "ReferenceDistribution",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    feature_importances = relationship(
        "FeatureImportance",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    snapshots = relationship(
        "Snapshot",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    drift_scores = relationship(
        "DriftScore",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    health_reports = relationship(
        "HealthReport",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    alerts = relationship(
        "Alert",
        back_populates="model",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
