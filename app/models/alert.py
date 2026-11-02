from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Alert(Base):
    """Active or resolved alerts emitted by drift monitoring."""

    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "feature_name",
            "window_date",
            "alert_type",
            name="uq_alert_model_feature_window_type",
        ),
        CheckConstraint(
            "alert_type IN ('drift_red','output_shift','data_gap')",
            name="ck_alerts_alert_type",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    model_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("models.id", ondelete="CASCADE"),
        nullable=False,
    )
    window_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    feature_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    alert_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str | None] = mapped_column(Text, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )

    model = relationship("ModelRegistry", back_populates="alerts")
