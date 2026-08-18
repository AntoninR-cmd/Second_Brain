"""Add resumable passage analysis, diagnostics, heartbeat, and metrics.

Revision ID: 20260818_0005
Revises: 20260817_0004
Create Date: 2026-08-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260818_0005"
down_revision: str | None = "20260817_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    null_passage_evidence = op.get_bind().scalar(
        sa.text("SELECT COUNT(*) FROM knowledge_evidence WHERE passage_id IS NULL")
    )
    if null_passage_evidence:
        raise RuntimeError(
            "Migration 0005 refusee: une preuve de connaissance sans passage doit "
            "etre reparee avant de garantir la tracabilite."
        )

    with op.batch_alter_table("knowledge_evidence", recreate="always") as batch_op:
        batch_op.alter_column(
            "passage_id",
            existing_type=sa.Uuid(),
            nullable=False,
        )

    with op.batch_alter_table("source_passages", recreate="always") as batch_op:
        batch_op.add_column(
            sa.Column(
                "analysis_status",
                sa.String(length=16),
                server_default=sa.text("'pending'"),
                nullable=False,
            )
        )
        batch_op.add_column(sa.Column("analysis_payload_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("analysis_error", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "analysis_attempt_count",
                sa.Integer(),
                server_default=sa.text("0"),
                nullable=False,
            )
        )
        batch_op.add_column(sa.Column("analysis_started_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("analysis_completed_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("analysis_last_activity_at", sa.DateTime(), nullable=True))
        for name, type_ in _passage_metric_columns():
            batch_op.add_column(sa.Column(name, type_, server_default=sa.text("0"), nullable=False))
        batch_op.create_check_constraint(
            "ck_source_passages_analysis_status",
            "analysis_status IN ('pending', 'running', 'completed', 'failed')",
        )
        batch_op.create_check_constraint(
            "ck_source_passages_analysis_attempt_count",
            "analysis_attempt_count >= 0",
        )
        batch_op.create_check_constraint(
            "ck_source_passages_analysis_metrics",
            "llm_call_count >= 0 AND llm_retry_count >= 0 "
            "AND llm_duration_ms >= 0 AND ollama_total_duration_ns >= 0 "
            "AND prompt_eval_count >= 0 AND prompt_eval_duration_ns >= 0 "
            "AND eval_count >= 0 AND eval_duration_ns >= 0 AND knowledge_count >= 0",
        )
        batch_op.create_check_constraint(
            "ck_source_passages_completed_payload",
            "analysis_status != 'completed' OR "
            "(analysis_payload_json IS NOT NULL AND intermediate_summary IS NOT NULL)",
        )

    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.add_column(sa.Column("error_code", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("error_type", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("error_detail", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("error_stage", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("error_passage_id", sa.Uuid(), nullable=True))
        batch_op.add_column(sa.Column("error_passage_index", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("error_attempt", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("error_call_type", sa.String(length=64), nullable=True))
        for name, type_ in _job_metric_columns():
            batch_op.add_column(sa.Column(name, type_, server_default=sa.text("0"), nullable=False))
        batch_op.create_foreign_key(
            "fk_processing_jobs_error_passage_id_source_passages",
            "source_passages",
            ["error_passage_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_processing_jobs_error_passage_id",
            ["error_passage_id"],
            unique=False,
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_error_passage_index",
            "error_passage_index IS NULL OR error_passage_index >= 0",
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_error_attempt",
            "error_attempt IS NULL OR error_attempt >= 0",
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_analysis_metrics",
            "llm_call_count >= 0 AND llm_retry_count >= 0 "
            "AND llm_duration_ms >= 0 AND ollama_total_duration_ns >= 0 "
            "AND prompt_eval_count >= 0 AND prompt_eval_duration_ns >= 0 "
            "AND eval_count >= 0 AND eval_duration_ns >= 0 AND knowledge_node_count >= 0",
        )


def downgrade() -> None:
    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_processing_jobs_analysis_metrics", type_="check")
        batch_op.drop_constraint("ck_processing_jobs_error_attempt", type_="check")
        batch_op.drop_constraint("ck_processing_jobs_error_passage_index", type_="check")
        batch_op.drop_index("ix_processing_jobs_error_passage_id")
        batch_op.drop_constraint(
            "fk_processing_jobs_error_passage_id_source_passages",
            type_="foreignkey",
        )
        for name, _type in reversed(_job_metric_columns()):
            batch_op.drop_column(name)
        batch_op.drop_column("error_call_type")
        batch_op.drop_column("error_attempt")
        batch_op.drop_column("error_passage_index")
        batch_op.drop_column("error_passage_id")
        batch_op.drop_column("error_stage")
        batch_op.drop_column("error_detail")
        batch_op.drop_column("error_type")
        batch_op.drop_column("error_code")

    with op.batch_alter_table("source_passages", recreate="always") as batch_op:
        batch_op.drop_constraint("ck_source_passages_completed_payload", type_="check")
        batch_op.drop_constraint("ck_source_passages_analysis_metrics", type_="check")
        batch_op.drop_constraint(
            "ck_source_passages_analysis_attempt_count",
            type_="check",
        )
        batch_op.drop_constraint("ck_source_passages_analysis_status", type_="check")
        for name, _type in reversed(_passage_metric_columns()):
            batch_op.drop_column(name)
        batch_op.drop_column("analysis_last_activity_at")
        batch_op.drop_column("analysis_completed_at")
        batch_op.drop_column("analysis_started_at")
        batch_op.drop_column("analysis_attempt_count")
        batch_op.drop_column("analysis_error")
        batch_op.drop_column("analysis_payload_json")
        batch_op.drop_column("analysis_status")

    with op.batch_alter_table("knowledge_evidence", recreate="always") as batch_op:
        batch_op.alter_column(
            "passage_id",
            existing_type=sa.Uuid(),
            nullable=True,
        )


def _passage_metric_columns() -> list[tuple[str, sa.types.TypeEngine]]:
    return [
        ("llm_call_count", sa.Integer()),
        ("llm_retry_count", sa.Integer()),
        ("llm_duration_ms", sa.BigInteger()),
        ("ollama_total_duration_ns", sa.BigInteger()),
        ("prompt_eval_count", sa.BigInteger()),
        ("prompt_eval_duration_ns", sa.BigInteger()),
        ("eval_count", sa.BigInteger()),
        ("eval_duration_ns", sa.BigInteger()),
        ("knowledge_count", sa.Integer()),
    ]


def _job_metric_columns() -> list[tuple[str, sa.types.TypeEngine]]:
    return [*_passage_metric_columns()[:-1], ("knowledge_node_count", sa.Integer())]
