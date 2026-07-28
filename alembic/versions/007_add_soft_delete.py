"""Add soft-delete support — deleted_at columns on Match, Prediction, Team.

Revision ID: 007
Revises: 006
Create Date: 2026-07-28

Changes
-------
1. Add nullable ``deleted_at`` timestamp column to ``matches`` table
2. Add nullable ``deleted_at`` timestamp column to ``predictions`` table
3. Add nullable ``deleted_at`` timestamp column to ``teams`` table

Rationale
---------
Hard-deleting rows from these foundational tables destroys historical
references needed for:
- Backtesting (deleted matches would break replay)
- Audit trails (model predictions must be preserved)
- Team lineage (renamed/disbanded teams still appear in past matches)

Instead, rows are soft-deleted by setting ``deleted_at`` to the current
timestamp. Active records have ``deleted_at IS NULL``.

All queries should filter on ``deleted_at IS NULL`` unless explicitly
retrieving deleted records for admin / forensic purposes.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. matches ──────────────────────────────────────
    op.add_column(
        "matches",
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Soft-delete timestamp. NULL = active record.",
        ),
    )
    op.create_index(
        "ix_matches_deleted_at",
        "matches",
        ["deleted_at"],
        postgresql_using="btree",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # ── 2. predictions ──────────────────────────────────
    op.add_column(
        "predictions",
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Soft-delete timestamp. NULL = active record.",
        ),
    )
    op.create_index(
        "ix_predictions_deleted_at",
        "predictions",
        ["deleted_at"],
        postgresql_using="btree",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # ── 3. teams ────────────────────────────────────────
    op.add_column(
        "teams",
        sa.Column(
            "deleted_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Soft-delete timestamp. NULL = active record.",
        ),
    )
    op.create_index(
        "ix_teams_deleted_at",
        "teams",
        ["deleted_at"],
        postgresql_using="btree",
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    # Drop partial indexes first
    op.drop_index("ix_matches_deleted_at", table_name="matches")
    op.drop_index("ix_predictions_deleted_at", table_name="predictions")
    op.drop_index("ix_teams_deleted_at", table_name="teams")

    # Drop columns
    op.drop_column("matches", "deleted_at")
    op.drop_column("predictions", "deleted_at")
    op.drop_column("teams", "deleted_at")
