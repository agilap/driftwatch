"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the initial DriftWatch schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "models",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "reference_distributions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column(
            "distribution", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_id", "feature_name", name="uq_refdist_model_feature"
        ),
    )

    op.create_table(
        "feature_importances",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("method", sa.Text(), nullable=True),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "method IN ('shap','coefficient','manual')",
            name="ck_feature_importance_method",
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_id", "feature_name", name="uq_feature_importance_model_feature"
        ),
    )

    op.create_table(
        "snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("window_date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column(
            "distribution", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_snapshots_model_id_window_date",
        "snapshots",
        ["model_id", "window_date"],
        unique=False,
    )

    op.create_table(
        "drift_scores",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("window_date", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.Text(), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=True),
        sa.Column("ks_pvalue", sa.Float(), nullable=True),
        sa.Column("psi", sa.Float(), nullable=True),
        sa.Column("js_divergence", sa.Float(), nullable=True),
        sa.Column("weighted_score", sa.Float(), nullable=True),
        sa.Column("severity", sa.Text(), nullable=True),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "severity IN ('green','yellow','red')", name="ck_drift_scores_severity"
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_drift_scores_model_id_window_date",
        "drift_scores",
        ["model_id", "window_date"],
        unique=False,
    )

    op.create_table(
        "health_reports",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("week_start", sa.Date(), nullable=False),
        sa.Column("week_end", sa.Date(), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column(
            "report_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("report_markdown", sa.Text(), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "alerts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("model_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("feature_name", sa.Text(), nullable=True),
        sa.Column("alert_type", sa.Text(), nullable=True),
        sa.Column("severity", sa.Text(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "alert_type IN ('drift_red','output_shift','data_gap')",
            name="ck_alerts_alert_type",
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop initial DriftWatch schema."""
    op.drop_table("alerts")
    op.drop_table("health_reports")
    op.drop_index("ix_drift_scores_model_id_window_date", table_name="drift_scores")
    op.drop_table("drift_scores")
    op.drop_index("ix_snapshots_model_id_window_date", table_name="snapshots")
    op.drop_table("snapshots")
    op.drop_table("feature_importances")
    op.drop_table("reference_distributions")
    op.drop_table("models")
