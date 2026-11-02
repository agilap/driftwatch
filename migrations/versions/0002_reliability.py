"""Enforce monitoring identities without silently deleting historical duplicates."""

from alembic import op
import sqlalchemy as sa

revision = "0002_reliability"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Constraint creation deliberately fails on legacy duplicates. Operators must
    # review them rather than allowing an upgrade to discard monitoring evidence.
    op.create_unique_constraint(
        "uq_snapshot_model_window_feature",
        "snapshots",
        ["model_id", "window_date", "feature_name"],
    )
    op.create_unique_constraint(
        "uq_drift_model_window_feature",
        "drift_scores",
        ["model_id", "window_date", "feature_name"],
    )
    op.create_unique_constraint(
        "uq_report_model_week",
        "health_reports",
        ["model_id", "week_start"],
    )
    op.add_column("alerts", sa.Column("window_date", sa.Date(), nullable=True))
    op.create_unique_constraint(
        "uq_alert_model_feature_window_type",
        "alerts",
        ["model_id", "feature_name", "window_date", "alert_type"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_alert_model_feature_window_type", "alerts", type_="unique")
    op.drop_column("alerts", "window_date")
    op.drop_constraint("uq_report_model_week", "health_reports", type_="unique")
    op.drop_constraint("uq_drift_model_window_feature", "drift_scores", type_="unique")
    op.drop_constraint("uq_snapshot_model_window_feature", "snapshots", type_="unique")
