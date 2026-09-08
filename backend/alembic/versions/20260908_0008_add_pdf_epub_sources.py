"""Add PDF/EPUB sources and document provenance.

Revision ID: 20260908_0008
Revises: 20260824_0007
Create Date: 2026-09-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260908_0008"
down_revision: str | None = "20260824_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _set_sqlite_foreign_keys(*, enabled: bool) -> None:
    """Toggle SQLite FK enforcement outside a transaction.

    SQLite recreates tables when a CHECK constraint changes. Disabling foreign
    keys prevents the temporary drop of ``sources`` from cascading into its
    Phase 2-6 child rows. Integrity is checked before enforcement is restored.
    """

    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    with op.get_context().autocommit_block():
        bind.exec_driver_sql(f"PRAGMA foreign_keys={'ON' if enabled else 'OFF'}")


def _restore_and_check_sqlite_foreign_keys() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return
    with op.get_context().autocommit_block():
        violations = bind.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
        bind.exec_driver_sql("PRAGMA foreign_keys=ON")
    if violations:
        details = ", ".join(
            f"{table}[{row_id}] -> {parent}" for table, row_id, parent, _ in violations[:5]
        )
        raise RuntimeError(f"Foreign-key violations after document migration: {details}")


def upgrade() -> None:
    _set_sqlite_foreign_keys(enabled=False)

    with op.batch_alter_table("sources", recreate="always") as batch_op:
        batch_op.drop_constraint("source_type", type_="check")
        batch_op.drop_constraint("processing_status", type_="check")
        batch_op.alter_column(
            "processing_status",
            existing_type=sa.String(length=5),
            type_=sa.String(length=9),
            existing_nullable=False,
            existing_server_default=sa.text("'ready'"),
        )
        batch_op.add_column(sa.Column("processing_error", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("page_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("chapter_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("language", sa.String(length=32), nullable=True))
        batch_op.create_check_constraint(
            "source_type",
            "type IN ('manual', 'srt', 'txt', 'pdf', 'epub')",
        )
        batch_op.create_check_constraint(
            "processing_status",
            "processing_status IN ('ready', 'needs_ocr')",
        )
        batch_op.create_check_constraint(
            "ck_sources_page_count",
            "page_count IS NULL OR page_count >= 0",
        )
        batch_op.create_check_constraint(
            "ck_sources_chapter_count",
            "chapter_count IS NULL OR chapter_count >= 0",
        )

    with op.batch_alter_table("source_segments", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("page_number", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("chapter_index", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("chapter_title", sa.String(length=512), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_source_segments_page_number",
            "page_number IS NULL OR page_number >= 1",
        )
        batch_op.create_check_constraint(
            "ck_source_segments_chapter_index",
            "chapter_index IS NULL OR chapter_index >= 0",
        )

    with op.batch_alter_table("knowledge_evidence", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("page_number", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("page_end_number", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("chapter_index", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("chapter_title", sa.String(length=512), nullable=True)
        )
        batch_op.create_check_constraint(
            "ck_knowledge_evidence_page_number",
            "page_number IS NULL OR page_number >= 1",
        )
        batch_op.create_check_constraint(
            "ck_knowledge_evidence_page_range",
            "page_end_number IS NULL OR page_number IS NULL "
            "OR page_end_number >= page_number",
        )
        batch_op.create_check_constraint(
            "ck_knowledge_evidence_chapter_index",
            "chapter_index IS NULL OR chapter_index >= 0",
        )

    _restore_and_check_sqlite_foreign_keys()


def downgrade() -> None:
    bind = op.get_bind()
    incompatible_count = bind.execute(
        sa.text(
            "SELECT count(*) FROM sources "
            "WHERE type IN ('pdf', 'epub') OR processing_status = 'needs_ocr'"
        )
    ).scalar_one()
    if incompatible_count:
        raise RuntimeError(
            "Cannot downgrade Phase 7 while PDF, EPUB, or OCR-required sources exist."
        )

    _set_sqlite_foreign_keys(enabled=False)

    with op.batch_alter_table("knowledge_evidence", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_knowledge_evidence_chapter_index", type_="check"
        )
        batch_op.drop_constraint("ck_knowledge_evidence_page_range", type_="check")
        batch_op.drop_constraint(
            "ck_knowledge_evidence_page_number", type_="check"
        )
        batch_op.drop_column("chapter_title")
        batch_op.drop_column("chapter_index")
        batch_op.drop_column("page_end_number")
        batch_op.drop_column("page_number")

    with op.batch_alter_table("source_segments", recreate="always") as batch_op:
        batch_op.drop_constraint(
            "ck_source_segments_chapter_index", type_="check"
        )
        batch_op.drop_constraint(
            "ck_source_segments_page_number", type_="check"
        )
        batch_op.drop_column("chapter_title")
        batch_op.drop_column("chapter_index")
        batch_op.drop_column("page_number")

    with op.batch_alter_table("sources", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_sources_chapter_count", type_="check")
        batch_op.drop_constraint("ck_sources_page_count", type_="check")
        batch_op.drop_constraint("processing_status", type_="check")
        batch_op.drop_constraint("source_type", type_="check")
        batch_op.drop_column("language")
        batch_op.drop_column("chapter_count")
        batch_op.drop_column("page_count")
        batch_op.drop_column("processing_error")
        batch_op.alter_column(
            "processing_status",
            existing_type=sa.String(length=9),
            type_=sa.String(length=5),
            existing_nullable=False,
            existing_server_default=sa.text("'ready'"),
        )
        batch_op.create_check_constraint(
            "source_type",
            "type IN ('manual', 'srt', 'txt')",
        )
        batch_op.create_check_constraint(
            "processing_status",
            "processing_status IN ('ready')",
        )

    _restore_and_check_sqlite_foreign_keys()
