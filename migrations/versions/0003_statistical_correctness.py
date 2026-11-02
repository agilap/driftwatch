"""Discard derived monitoring results computed with incorrect statistical formulas."""

from alembic import op

revision = "0003_statistical_correctness"
down_revision = "0002_reliability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Legacy snapshots have independent bins; analysis returns 409 until re-ingestion.
    op.execute("DELETE FROM drift_scores")
    op.execute("DELETE FROM health_reports")


def downgrade() -> None:
    # Deleted derived results cannot be restored by downgrading.
    pass
