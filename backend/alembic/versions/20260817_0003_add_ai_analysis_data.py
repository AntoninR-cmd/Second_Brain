"""Add passages, knowledge, evidence, tags, and analysis jobs.

Revision ID: 20260817_0003
Revises: 20260817_0002
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260817_0003"
down_revision: str | None = "20260817_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column(
        "sources",
        sa.Column(
            "analysis_status",
            sa.String(length=12),
            sa.CheckConstraint(
                "analysis_status IN ('not_analyzed', 'queued', 'processing', 'analyzed', 'error')",
                name="ck_sources_analysis_status",
            ),
            server_default=sa.text("'not_analyzed'"),
            nullable=False,
        ),
    )
    op.add_column("sources", sa.Column("analysis_error", sa.Text(), nullable=True))
    op.add_column("sources", sa.Column("analysis_started_at", sa.DateTime(), nullable=True))
    op.add_column("sources", sa.Column("analysis_completed_at", sa.DateTime(), nullable=True))
    op.create_index(
        op.f("ix_sources_analysis_status"),
        "sources",
        ["analysis_status"],
        unique=False,
    )

    op.create_table(
        "source_passages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("passage_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("first_segment_index", sa.Integer(), nullable=True),
        sa.Column("last_segment_index", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("intermediate_summary", sa.Text(), nullable=True),
        sa.CheckConstraint("token_count >= 0", name="ck_source_passages_token_count"),
        sa.CheckConstraint(
            "first_segment_index IS NULL OR last_segment_index IS NULL "
            "OR first_segment_index <= last_segment_index",
            name="ck_source_passages_segment_range",
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_end IS NULL OR char_start <= char_end",
            name="ck_source_passages_char_range",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id",
            "passage_index",
            name="uq_source_passages_source_id_index",
        ),
    )
    op.create_index(
        op.f("ix_source_passages_source_id"),
        "source_passages",
        ["source_id"],
        unique=False,
    )

    op.create_table(
        "source_passage_segments",
        sa.Column("passage_id", sa.Uuid(), nullable=False),
        sa.Column("segment_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.CheckConstraint("position >= 0", name="ck_source_passage_segments_position"),
        sa.ForeignKeyConstraint(
            ["passage_id"],
            ["source_passages.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["segment_id"],
            ["source_segments.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("passage_id", "segment_id"),
        sa.UniqueConstraint(
            "passage_id",
            "position",
            name="uq_source_passage_segments_passage_position",
        ),
    )
    op.create_index(
        op.f("ix_source_passage_segments_segment_id"),
        "source_passage_segments",
        ["segment_id"],
        unique=False,
    )

    op.create_table(
        "knowledge_nodes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
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
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_knowledge_nodes_source_id"),
        "knowledge_nodes",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_nodes_created_at"),
        "knowledge_nodes",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "tags",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("normalized_name", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", name="uq_tags_normalized_name"),
    )

    op.create_table(
        "knowledge_node_tags",
        sa.Column("knowledge_node_id", sa.Uuid(), nullable=False),
        sa.Column("tag_id", sa.Uuid(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["knowledge_node_id"],
            ["knowledge_nodes.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["tag_id"], ["tags.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("knowledge_node_id", "tag_id"),
    )
    op.create_index(
        op.f("ix_knowledge_node_tags_tag_id"),
        "knowledge_node_tags",
        ["tag_id"],
        unique=False,
    )

    op.create_table(
        "knowledge_evidence",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("knowledge_node_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("passage_id", sa.Uuid(), nullable=True),
        sa.Column("evidence_index", sa.Integer(), nullable=False),
        sa.Column("first_segment_id", sa.Uuid(), nullable=True),
        sa.Column("last_segment_id", sa.Uuid(), nullable=True),
        sa.Column("original_excerpt", sa.Text(), nullable=False),
        sa.Column("start_ms", sa.BigInteger(), nullable=True),
        sa.Column("end_ms", sa.BigInteger(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "start_ms IS NULL OR end_ms IS NULL OR start_ms <= end_ms",
            name="ck_knowledge_evidence_time_range",
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_end IS NULL OR char_start <= char_end",
            name="ck_knowledge_evidence_char_range",
        ),
        sa.ForeignKeyConstraint(
            ["first_segment_id"],
            ["source_segments.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_node_id"],
            ["knowledge_nodes.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["last_segment_id"],
            ["source_segments.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["passage_id"],
            ["source_passages.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "knowledge_node_id",
            "evidence_index",
            name="uq_knowledge_evidence_node_index",
        ),
    )
    op.create_index(
        op.f("ix_knowledge_evidence_knowledge_node_id"),
        "knowledge_evidence",
        ["knowledge_node_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_evidence_source_id"),
        "knowledge_evidence",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_knowledge_evidence_passage_id"),
        "knowledge_evidence",
        ["passage_id"],
        unique=False,
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column(
            "kind",
            sa.Enum(
                "analyze_source",
                name="processing_job_kind",
                native_enum=False,
                create_constraint=True,
            ),
            server_default=sa.text("'analyze_source'"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "succeeded",
                "failed",
                name="processing_job_status",
                native_enum=False,
                create_constraint=True,
            ),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("stage", sa.String(length=64), nullable=True),
        sa.Column("progress_current", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("progress_total", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("progress_message", sa.String(length=512), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
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
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint("progress_current >= 0", name="ck_processing_jobs_progress_current"),
        sa.CheckConstraint("progress_total >= 0", name="ck_processing_jobs_progress_total"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_processing_jobs_attempt_count"),
        sa.ForeignKeyConstraint(["source_id"], ["sources.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_processing_jobs_source_id"),
        "processing_jobs",
        ["source_id"],
        unique=False,
    )
    op.create_index(
        "ix_processing_jobs_status_created_at",
        "processing_jobs",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_processing_jobs_status_created_at", table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_source_id"), table_name="processing_jobs")
    op.drop_table("processing_jobs")

    op.drop_index(op.f("ix_knowledge_evidence_passage_id"), table_name="knowledge_evidence")
    op.drop_index(op.f("ix_knowledge_evidence_source_id"), table_name="knowledge_evidence")
    op.drop_index(
        op.f("ix_knowledge_evidence_knowledge_node_id"),
        table_name="knowledge_evidence",
    )
    op.drop_table("knowledge_evidence")

    op.drop_index(op.f("ix_knowledge_node_tags_tag_id"), table_name="knowledge_node_tags")
    op.drop_table("knowledge_node_tags")
    op.drop_table("tags")

    op.drop_index(op.f("ix_knowledge_nodes_created_at"), table_name="knowledge_nodes")
    op.drop_index(op.f("ix_knowledge_nodes_source_id"), table_name="knowledge_nodes")
    op.drop_table("knowledge_nodes")

    op.drop_index(
        op.f("ix_source_passage_segments_segment_id"),
        table_name="source_passage_segments",
    )
    op.drop_table("source_passage_segments")
    op.drop_index(op.f("ix_source_passages_source_id"), table_name="source_passages")
    op.drop_table("source_passages")

    op.drop_index(op.f("ix_sources_analysis_status"), table_name="sources")
    op.drop_column("sources", "analysis_completed_at")
    op.drop_column("sources", "analysis_started_at")
    op.drop_column("sources", "analysis_error")
    op.drop_column("sources", "analysis_status")
    op.drop_column("sources", "summary")
