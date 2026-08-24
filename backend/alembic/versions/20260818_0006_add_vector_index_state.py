"""Add versioned embedding profiles and resumable vector-index state.

Revision ID: 20260818_0006
Revises: 20260818_0005
Create Date: 2026-08-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260818_0006"
down_revision: str | None = "20260818_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embedding_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "provider",
            sa.String(length=64),
            server_default=sa.text("'ollama'"),
            nullable=False,
        ),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("model_digest", sa.String(length=128), nullable=True),
        sa.Column("dimensions", sa.Integer(), nullable=True),
        sa.Column(
            "distance",
            sa.String(length=16),
            server_default=sa.text("'cosine'"),
            nullable=False,
        ),
        sa.Column("collection_name", sa.String(length=255), nullable=False),
        sa.Column("semantic_text_version", sa.String(length=64), nullable=False),
        sa.Column("logical_generation", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'building'"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "length(trim(provider)) > 0",
            name="ck_embedding_profiles_provider",
        ),
        sa.CheckConstraint(
            "length(trim(model_name)) > 0",
            name="ck_embedding_profiles_model",
        ),
        sa.CheckConstraint(
            "dimensions IS NULL OR dimensions > 0",
            name="ck_embedding_profiles_dimensions",
        ),
        sa.CheckConstraint(
            "logical_generation > 0",
            name="ck_embedding_profiles_logical_generation",
        ),
        sa.CheckConstraint(
            "distance IN ('cosine')",
            name="ck_embedding_profiles_distance",
        ),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'retired', 'failed')",
            name="ck_embedding_profiles_status",
        ),
        sa.CheckConstraint(
            "status != 'active' OR (dimensions IS NOT NULL AND activated_at IS NOT NULL)",
            name="ck_embedding_profiles_active_metadata",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "logical_generation",
            name="uq_embedding_profiles_logical_generation",
        ),
        sa.UniqueConstraint(
            "collection_name",
            name="uq_embedding_profiles_collection_name",
        ),
    )
    op.create_index(
        "uq_embedding_profiles_single_active",
        "embedding_profiles",
        ["status"],
        unique=True,
        sqlite_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "knowledge_embeddings",
        sa.Column("knowledge_node_id", sa.Uuid(), nullable=False),
        sa.Column("embedding_profile_id", sa.Uuid(), nullable=False),
        sa.Column("text_fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "length(text_fingerprint) = 64",
            name="ck_knowledge_embeddings_fingerprint",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'indexed', 'stale', 'failed')",
            name="ck_knowledge_embeddings_status",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="ck_knowledge_embeddings_attempt_count",
        ),
        sa.CheckConstraint(
            "status != 'indexed' OR indexed_at IS NOT NULL",
            name="ck_knowledge_embeddings_indexed_at",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_profile_id"],
            ["embedding_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_node_id"],
            ["knowledge_nodes.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("knowledge_node_id", "embedding_profile_id"),
    )
    op.create_index(
        "ix_knowledge_embeddings_profile_status",
        "knowledge_embeddings",
        ["embedding_profile_id", "status"],
        unique=False,
    )

    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint("processing_job_kind", type_="check")
        batch_op.alter_column(
            "kind",
            existing_type=sa.String(length=14),
            type_=sa.String(length=20),
            existing_nullable=False,
            existing_server_default=sa.text("'analyze_source'"),
        )
        batch_op.alter_column(
            "source_id",
            existing_type=sa.Uuid(),
            nullable=True,
        )
        batch_op.add_column(sa.Column("embedding_profile_id", sa.Uuid(), nullable=True))
        for name, type_ in _embedding_metric_columns():
            batch_op.add_column(sa.Column(name, type_, server_default=sa.text("0"), nullable=False))
        batch_op.create_check_constraint(
            "processing_job_kind",
            "kind IN ('analyze_source', 'index_knowledge', 'rebuild_vector_index')",
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_analysis_source",
            "kind != 'analyze_source' OR source_id IS NOT NULL",
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_embedding_metrics",
            "embedding_batch_count >= 0 AND embedding_item_count >= 0 "
            "AND embedding_duration_ms >= 0 AND embedding_total_duration_ns >= 0 "
            "AND embedding_prompt_eval_count >= 0",
        )
        batch_op.create_foreign_key(
            "fk_processing_jobs_embedding_profile_id",
            "embedding_profiles",
            ["embedding_profile_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_processing_jobs_embedding_profile_id",
            ["embedding_profile_id"],
            unique=False,
        )


def downgrade() -> None:
    op.execute("DELETE FROM processing_jobs WHERE kind != 'analyze_source'")

    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.drop_index("ix_processing_jobs_embedding_profile_id")
        batch_op.drop_constraint(
            "fk_processing_jobs_embedding_profile_id",
            type_="foreignkey",
        )
        batch_op.drop_constraint("ck_processing_jobs_embedding_metrics", type_="check")
        batch_op.drop_constraint("ck_processing_jobs_analysis_source", type_="check")
        batch_op.drop_constraint("processing_job_kind", type_="check")
        for name, _type in reversed(_embedding_metric_columns()):
            batch_op.drop_column(name)
        batch_op.drop_column("embedding_profile_id")
        batch_op.alter_column(
            "source_id",
            existing_type=sa.Uuid(),
            nullable=False,
        )
        batch_op.alter_column(
            "kind",
            existing_type=sa.String(length=20),
            type_=sa.String(length=14),
            existing_nullable=False,
            existing_server_default=sa.text("'analyze_source'"),
        )
        batch_op.create_check_constraint(
            "processing_job_kind",
            "kind IN ('analyze_source')",
        )

    op.drop_index(
        "ix_knowledge_embeddings_profile_status",
        table_name="knowledge_embeddings",
    )
    op.drop_table("knowledge_embeddings")
    op.drop_index(
        "uq_embedding_profiles_single_active",
        table_name="embedding_profiles",
    )
    op.drop_table("embedding_profiles")


def _embedding_metric_columns() -> list[tuple[str, sa.types.TypeEngine]]:
    return [
        ("embedding_batch_count", sa.Integer()),
        ("embedding_item_count", sa.Integer()),
        ("embedding_duration_ms", sa.BigInteger()),
        ("embedding_total_duration_ns", sa.BigInteger()),
        ("embedding_prompt_eval_count", sa.BigInteger()),
    ]
