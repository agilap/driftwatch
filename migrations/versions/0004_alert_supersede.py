"""Let new data supersede a window's alerts while keeping them as audit records."""

from alembic import op
import sqlalchemy as sa

revision = "0004_alert_supersede"
down_revision = "0003_statistical_correctness"
branch_labels = None
depends_on = None

COLUMNS = ["model_id", "feature_name", "window_date", "alert_type"]


def upgrade() -> None:
    op.add_column(
        "alerts", sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.drop_constraint("uq_alert_model_feature_window_type", "alerts", type_="unique")
    op.create_index(
        "uq_alert_model_feature_window_type",
        "alerts",
        COLUMNS,
        unique=True,
        postgresql_where=sa.text("superseded_at IS NULL"),
    )


def downgrade() -> None:
    # The old constraint allows one row per window, so superseded rows cannot remain.
    op.execute("DELETE FROM alerts WHERE superseded_at IS NOT NULL")
    op.drop_index("uq_alert_model_feature_window_type", table_name="alerts")
    op.create_unique_constraint("uq_alert_model_feature_window_type", "alerts", COLUMNS)
    op.drop_column("alerts", "superseded_at")
