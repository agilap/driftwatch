from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, Date, DateTime, Float, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DriftScore(Base):
    """Computed feature drift scores for a model and time window."""

    __tablename__ = "drift_scores"
    __table_args__ = (
        CheckConstraint("severity IN ('green','yellow','red')", name="ck_drift_scores_severity"),
        Index("ix_drift_scores_model_id_window_date", "model_id", "window_date"),
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
    window_date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_name: Mapped[str] = mapped_column(Text, nullable=False)
    ks_statistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    ks_pvalue: Mapped[float | None] = mapped_column(Float, nullable=True)
    psi: Mapped[float | None] = mapped_column(Float, nullable=True)
    js_divergence: Mapped[float | None] = mapped_column(Float, nullable=True)
    weighted_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    severity: Mapped[str | None] = mapped_column(Text, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )

    model = relationship("ModelRegistry", back_populates="drift_scores")
