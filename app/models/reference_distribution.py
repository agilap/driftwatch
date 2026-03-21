from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ReferenceDistribution(Base):
    """Reference (training) feature distribution per model."""

    __tablename__ = "reference_distributions"
    __table_args__ = (
        UniqueConstraint("model_id", "feature_name", name="uq_refdist_model_feature"),
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
    feature_name: Mapped[str] = mapped_column(Text, nullable=False)
    distribution: Mapped[dict] = mapped_column(JSONB, nullable=False)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        nullable=False,
    )

    model = relationship("ModelRegistry", back_populates="reference_distributions")
