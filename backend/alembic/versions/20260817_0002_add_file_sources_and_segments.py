"""Add TXT/SRT metadata and source segments.

Revision ID: 20260817_0002
Revises: 20260816_0001
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_0002"
down_revision: str | None = "20260816_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _phase_two_sources_table(name: str) -> None:
    op.create_table(
        name,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "manual",
                "srt",
                "txt",
                name="source_type",
                native_enum=False,
                create_constraint=True,
            ),
            server_default=sa.text("'manual'"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("original_file_path", sa.String(length=512), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column(
            "processing_status",
            sa.Enum(
                "ready",
                name="processing_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default=sa.text("'ready'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def _phase_one_sources_table(name: str) -> None:
    op.create_table(
        name,
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "type",
            sa.Enum(
                "manual",
                name="source_type",
                native_enum=False,
                create_constraint=True,
            ),
            server_default=sa.text("'manual'"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column(
            "processing_status",
            sa.Enum(
                "ready",
                name="processing_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default=sa.text("'ready'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def upgrade() -> None:
    # SQLite cannot alter the CHECK generated for an Enum. Recreate the table
    # explicitly so existing Phase 1 rows are retained while the type expands.
    _phase_two_sources_table("sources_phase2")
    op.execute(
        """
        INSERT INTO sources_phase2 (
            id, type, title, author, raw_text, processing_status, created_at, updated_at
        )
        SELECT
            id, type, title, author, raw_text, processing_status, created_at, updated_at
        FROM sources
        """
    )
    op.drop_index(op.f("ix_sources_created_at"), table_name="sources")
    op.drop_table("sources")
    op.rename_table("sources_phase2", "sources")
    op.create_index(op.f("ix_sources_created_at"), "sources", ["created_at"], unique=False)
    op.create_index(op.f("ix_sources_file_sha256"), "sources", ["file_sha256"], unique=False)

    op.create_table(
        "source_segments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=True),
        sa.Column("end_ms", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id",
            "segment_index",
            name="uq_source_segments_source_id_index",
        ),
    )
    op.create_index(
        op.f("ix_source_segments_source_id"),
        "source_segments",
        ["source_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_source_segments_source_id"), table_name="source_segments")
    op.drop_table("source_segments")

    _phase_one_sources_table("sources_phase1")
    op.execute(
        """
        INSERT INTO sources_phase1 (
            id, type, title, author, raw_text, processing_status, created_at, updated_at
        )
        SELECT
            id, type, title, author, raw_text, processing_status, created_at, updated_at
        FROM sources
        WHERE type = 'manual'
        """
    )
    op.drop_index(op.f("ix_sources_file_sha256"), table_name="sources")
    op.drop_index(op.f("ix_sources_created_at"), table_name="sources")
    op.drop_table("sources")
    op.rename_table("sources_phase1", "sources")
    op.create_index(op.f("ix_sources_created_at"), "sources", ["created_at"], unique=False)
