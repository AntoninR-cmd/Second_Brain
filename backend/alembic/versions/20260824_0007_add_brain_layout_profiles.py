"""Add versioned, reconstructible mathematical brain layouts.

Revision ID: 20260824_0007
Revises: 20260818_0006
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260824_0007"
down_revision: str | None = "20260818_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "brain_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("embedding_profile_id", sa.Uuid(), nullable=True),
        sa.Column("embedding_provider", sa.String(length=64), nullable=False),
        sa.Column("embedding_model_name", sa.String(length=255), nullable=False),
        sa.Column("embedding_model_digest", sa.String(length=128), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding_semantic_text_version", sa.String(length=64), nullable=False),
        sa.Column("embedding_logical_generation", sa.Integer(), nullable=False),
        sa.Column("input_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column(
            "parameters_json",
            sa.Text(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("parameters_digest", sa.String(length=64), nullable=False),
        sa.Column("logical_generation", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'building'"),
            nullable=False,
        ),
        sa.Column(
            "knowledge_node_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "cluster_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "edge_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "unassigned_node_count",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "statistics_json",
            sa.Text(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "relations_duration_ms",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "clustering_duration_ms",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "umap_duration_ms",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "labeling_duration_ms",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "total_duration_ms",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "label_strategy",
            sa.String(length=16),
            server_default=sa.text("'deterministic'"),
            nullable=False,
        ),
        sa.Column("label_model_name", sa.String(length=255), nullable=True),
        sa.Column("label_model_digest", sa.String(length=128), nullable=True),
        sa.Column("labels_generated_at", sa.DateTime(), nullable=True),
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
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "length(trim(embedding_provider)) > 0",
            name="ck_brain_profiles_embedding_provider",
        ),
        sa.CheckConstraint(
            "length(trim(embedding_model_name)) > 0",
            name="ck_brain_profiles_embedding_model_name",
        ),
        sa.CheckConstraint(
            "embedding_dimensions > 0",
            name="ck_brain_profiles_embedding_dimensions",
        ),
        sa.CheckConstraint(
            "embedding_logical_generation > 0",
            name="ck_brain_profiles_embedding_generation",
        ),
        sa.CheckConstraint(
            "length(trim(embedding_semantic_text_version)) > 0",
            name="ck_brain_profiles_semantic_text_version",
        ),
        sa.CheckConstraint(
            "length(input_fingerprint) = 64",
            name="ck_brain_profiles_input_fingerprint",
        ),
        sa.CheckConstraint(
            "length(trim(algorithm_version)) > 0",
            name="ck_brain_profiles_algorithm_version",
        ),
        sa.CheckConstraint(
            "length(parameters_digest) = 64",
            name="ck_brain_profiles_parameters_digest",
        ),
        sa.CheckConstraint(
            "logical_generation > 0",
            name="ck_brain_profiles_logical_generation",
        ),
        sa.CheckConstraint(
            "status IN ('building', 'ready', 'stale', 'error')",
            name="ck_brain_profiles_status",
        ),
        sa.CheckConstraint(
            "knowledge_node_count >= 0 AND cluster_count >= 0 AND edge_count >= 0 "
            "AND unassigned_node_count >= 0",
            name="ck_brain_profiles_counts",
        ),
        sa.CheckConstraint(
            "relations_duration_ms >= 0 AND clustering_duration_ms >= 0 "
            "AND umap_duration_ms >= 0 AND labeling_duration_ms >= 0 "
            "AND total_duration_ms >= 0",
            name="ck_brain_profiles_durations",
        ),
        sa.CheckConstraint(
            "label_strategy IN ('deterministic', 'ollama', 'mixed')",
            name="ck_brain_profiles_label_strategy",
        ),
        sa.CheckConstraint(
            "status != 'ready' OR (completed_at IS NOT NULL AND activated_at IS NOT NULL)",
            name="ck_brain_profiles_ready_dates",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_profile_id"],
            ["embedding_profiles.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "logical_generation",
            name="uq_brain_profiles_logical_generation",
        ),
    )
    op.create_index(
        "ix_brain_profiles_embedding_profile_id",
        "brain_profiles",
        ["embedding_profile_id"],
        unique=False,
    )
    op.create_index(
        "uq_brain_profiles_single_ready",
        "brain_profiles",
        ["status"],
        unique=True,
        sqlite_where=sa.text("status = 'ready'"),
    )
    op.create_index(
        "uq_brain_profiles_single_building",
        "brain_profiles",
        ["status"],
        unique=True,
        sqlite_where=sa.text("status = 'building'"),
    )

    op.create_table(
        "brain_clusters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("brain_profile_id", sa.Uuid(), nullable=False),
        sa.Column("parent_cluster_id", sa.Uuid(), nullable=True),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "label_source",
            sa.String(length=16),
            server_default=sa.text("'deterministic'"),
            nullable=False,
        ),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("centroid_json", sa.Text(), nullable=False),
        sa.Column(
            "representative_nodes_json",
            sa.Text(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint("level >= 0", name="ck_brain_clusters_level"),
        sa.CheckConstraint(
            "length(trim(label)) > 0",
            name="ck_brain_clusters_label",
        ),
        sa.CheckConstraint(
            "label_source IN ('deterministic', 'ollama')",
            name="ck_brain_clusters_label_source",
        ),
        sa.CheckConstraint(
            "member_count > 0",
            name="ck_brain_clusters_member_count",
        ),
        sa.CheckConstraint(
            "x >= -1.0 AND x <= 1.0 AND y >= -1.0 AND y <= 1.0",
            name="ck_brain_clusters_coordinates",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id"],
            ["brain_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id", "parent_cluster_id"],
            ["brain_clusters.brain_profile_id", "brain_clusters.id"],
            name="fk_brain_clusters_parent_same_profile",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", "brain_profile_id"),
        sa.UniqueConstraint(
            "brain_profile_id",
            "id",
            name="uq_brain_clusters_profile_id",
        ),
    )
    op.create_index(
        "ix_brain_clusters_profile_level",
        "brain_clusters",
        ["brain_profile_id", "level"],
        unique=False,
    )
    op.create_index(
        "ix_brain_clusters_profile_parent",
        "brain_clusters",
        ["brain_profile_id", "parent_cluster_id"],
        unique=False,
    )

    op.create_table(
        "brain_node_layouts",
        sa.Column("brain_profile_id", sa.Uuid(), nullable=False),
        sa.Column("knowledge_node_id", sa.Uuid(), nullable=False),
        sa.Column("cluster_id", sa.Uuid(), nullable=True),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column(
            "is_unassigned",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("membership_confidence", sa.Float(), nullable=True),
        sa.Column("representative_rank", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "x >= -1.0 AND x <= 1.0 AND y >= -1.0 AND y <= 1.0",
            name="ck_brain_node_layouts_coordinates",
        ),
        sa.CheckConstraint(
            "(is_unassigned = 1 AND cluster_id IS NULL) "
            "OR (is_unassigned = 0 AND cluster_id IS NOT NULL)",
            name="ck_brain_node_layouts_assignment",
        ),
        sa.CheckConstraint(
            "is_unassigned IN (0, 1)",
            name="ck_brain_node_layouts_is_unassigned",
        ),
        sa.CheckConstraint(
            "membership_confidence IS NULL "
            "OR (membership_confidence >= 0.0 AND membership_confidence <= 1.0)",
            name="ck_brain_node_layouts_confidence",
        ),
        sa.CheckConstraint(
            "representative_rank IS NULL OR representative_rank > 0",
            name="ck_brain_node_layouts_representative_rank",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id"],
            ["brain_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_node_id"],
            ["knowledge_nodes.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id", "cluster_id"],
            ["brain_clusters.brain_profile_id", "brain_clusters.id"],
            name="fk_brain_node_layouts_cluster_same_profile",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("brain_profile_id", "knowledge_node_id"),
    )
    op.create_index(
        "ix_brain_node_layouts_profile_cluster",
        "brain_node_layouts",
        ["brain_profile_id", "cluster_id"],
        unique=False,
    )

    op.create_table(
        "brain_edges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("brain_profile_id", sa.Uuid(), nullable=False),
        sa.Column("source_node_id", sa.Uuid(), nullable=False),
        sa.Column("target_node_id", sa.Uuid(), nullable=False),
        sa.Column("cosine_score", sa.Float(), nullable=False),
        sa.Column(
            "tag_bonus",
            sa.Float(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column(
            "is_mutual",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("source_rank", sa.Integer(), nullable=True),
        sa.Column("target_rank", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source_node_id != target_node_id",
            name="ck_brain_edges_distinct_nodes",
        ),
        sa.CheckConstraint(
            "source_node_id < target_node_id",
            name="ck_brain_edges_canonical_order",
        ),
        sa.CheckConstraint(
            "cosine_score >= -1.0 AND cosine_score <= 1.0",
            name="ck_brain_edges_cosine_score",
        ),
        sa.CheckConstraint("tag_bonus >= 0.0", name="ck_brain_edges_tag_bonus"),
        sa.CheckConstraint(
            "source_rank IS NULL OR source_rank > 0",
            name="ck_brain_edges_source_rank",
        ),
        sa.CheckConstraint(
            "target_rank IS NULL OR target_rank > 0",
            name="ck_brain_edges_target_rank",
        ),
        sa.CheckConstraint(
            "is_mutual IN (0, 1)",
            name="ck_brain_edges_is_mutual",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id"],
            ["brain_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id", "source_node_id"],
            ["brain_node_layouts.brain_profile_id", "brain_node_layouts.knowledge_node_id"],
            name="fk_brain_edges_source_layout",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["brain_profile_id", "target_node_id"],
            ["brain_node_layouts.brain_profile_id", "brain_node_layouts.knowledge_node_id"],
            name="fk_brain_edges_target_layout",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "brain_profile_id",
            "source_node_id",
            "target_node_id",
            name="uq_brain_edges_profile_nodes",
        ),
    )
    op.create_index(
        "ix_brain_edges_profile_score",
        "brain_edges",
        ["brain_profile_id", "final_score"],
        unique=False,
    )
    op.create_index(
        "ix_brain_edges_profile_source",
        "brain_edges",
        ["brain_profile_id", "source_node_id"],
        unique=False,
    )
    op.create_index(
        "ix_brain_edges_profile_target",
        "brain_edges",
        ["brain_profile_id", "target_node_id"],
        unique=False,
    )

    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.drop_constraint("processing_job_kind", type_="check")
        batch_op.add_column(sa.Column("brain_profile_id", sa.Uuid(), nullable=True))
        batch_op.create_check_constraint(
            "processing_job_kind",
            "kind IN ('analyze_source', 'index_knowledge', 'rebuild_vector_index', "
            "'build_brain', 'relabel_brain')",
        )
        batch_op.create_check_constraint(
            "ck_processing_jobs_brain_profile",
            "kind NOT IN ('build_brain', 'relabel_brain') OR brain_profile_id IS NOT NULL",
        )
        batch_op.create_foreign_key(
            "fk_processing_jobs_brain_profile_id",
            "brain_profiles",
            ["brain_profile_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_processing_jobs_brain_profile_id",
            ["brain_profile_id"],
            unique=False,
        )


def downgrade() -> None:
    op.execute("DELETE FROM processing_jobs WHERE kind IN ('build_brain', 'relabel_brain')")

    with op.batch_alter_table("processing_jobs", recreate="always") as batch_op:
        batch_op.drop_index("ix_processing_jobs_brain_profile_id")
        batch_op.drop_constraint(
            "fk_processing_jobs_brain_profile_id",
            type_="foreignkey",
        )
        batch_op.drop_constraint("ck_processing_jobs_brain_profile", type_="check")
        batch_op.drop_constraint("processing_job_kind", type_="check")
        batch_op.drop_column("brain_profile_id")
        batch_op.create_check_constraint(
            "processing_job_kind",
            "kind IN ('analyze_source', 'index_knowledge', 'rebuild_vector_index')",
        )

    op.drop_index("ix_brain_edges_profile_target", table_name="brain_edges")
    op.drop_index("ix_brain_edges_profile_source", table_name="brain_edges")
    op.drop_index("ix_brain_edges_profile_score", table_name="brain_edges")
    op.drop_table("brain_edges")
    op.drop_index(
        "ix_brain_node_layouts_profile_cluster",
        table_name="brain_node_layouts",
    )
    op.drop_table("brain_node_layouts")
    op.drop_index("ix_brain_clusters_profile_parent", table_name="brain_clusters")
    op.drop_index("ix_brain_clusters_profile_level", table_name="brain_clusters")
    op.drop_table("brain_clusters")
    op.drop_index("uq_brain_profiles_single_building", table_name="brain_profiles")
    op.drop_index("uq_brain_profiles_single_ready", table_name="brain_profiles")
    op.drop_index("ix_brain_profiles_embedding_profile_id", table_name="brain_profiles")
    op.drop_table("brain_profiles")
