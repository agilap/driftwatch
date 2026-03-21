"""initial driftwatch scaffold

Revision ID: 0001_initial
Revises:
Create Date: 2026-03-22 00:00:00.000000

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial revision placeholder for upcoming ORM tables."""


def downgrade() -> None:
    """Rollback initial revision placeholder."""
