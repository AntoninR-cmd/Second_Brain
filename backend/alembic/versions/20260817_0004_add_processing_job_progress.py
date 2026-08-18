"""Add persistent progress details to processing jobs.

Revision ID: 20260817_0004
Revises: 20260817_0003
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_0004"
down_revision: str | None = "20260817_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column(
                "progress_percent",
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "last_activity_at",
                sa.DateTime(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            )
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_progress_percent",
            "progress_percent >= 0 AND progress_percent <= 100",
        )
    op.execute(
        "UPDATE processing_jobs "
        "SET last_activity_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)"
    )


def downgrade() -> None:
    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_processing_jobs_progress_percent",
            type_="check",
        )
        batch_op.drop_column("last_activity_at")
        batch_op.drop_column("progress_percent")
