from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from second_brain.db.base import Base, UTCDateTime, utc_now

if TYPE_CHECKING:
    from second_brain.db.models.embedding import EmbeddingProfile
    from second_brain.db.models.knowledge import KnowledgeNode
    from second_brain.db.models.processing import ProcessingJob


class BrainProfileStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
    STALE = "stale"
    ERROR = "error"


class BrainLabelSource(str, Enum):
    DETERMINISTIC = "deterministic"
    OLLAMA = "ollama"


class BrainLabelStrategy(str, Enum):
    DETERMINISTIC = "deterministic"
    OLLAMA = "ollama"
    MIXED = "mixed"


class BrainProfile(Base):
    """One immutable mathematical snapshot of the active semantic memory."""

    __tablename__ = "brain_profiles"
    __table_args__ = (
        CheckConstraint(
            "length(trim(embedding_provider)) > 0",
            name="ck_brain_profiles_embedding_provider",
        ),
        CheckConstraint(
            "length(trim(embedding_model_name)) > 0",
            name="ck_brain_profiles_embedding_model_name",
        ),
        CheckConstraint(
            "embedding_dimensions > 0",
            name="ck_brain_profiles_embedding_dimensions",
        ),
        CheckConstraint(
            "embedding_logical_generation > 0",
            name="ck_brain_profiles_embedding_generation",
        ),
        CheckConstraint(
            "length(trim(embedding_semantic_text_version)) > 0",
            name="ck_brain_profiles_semantic_text_version",
        ),
        CheckConstraint(
            "length(input_fingerprint) = 64",
            name="ck_brain_profiles_input_fingerprint",
        ),
        CheckConstraint(
            "length(trim(algorithm_version)) > 0",
            name="ck_brain_profiles_algorithm_version",
        ),
        CheckConstraint(
            "length(parameters_digest) = 64",
            name="ck_brain_profiles_parameters_digest",
        ),
        CheckConstraint(
            "logical_generation > 0",
            name="ck_brain_profiles_logical_generation",
        ),
        CheckConstraint(
            "status IN ('building', 'ready', 'stale', 'error')",
            name="ck_brain_profiles_status",
        ),
        CheckConstraint(
            "knowledge_node_count >= 0 AND cluster_count >= 0 AND edge_count >= 0 "
            "AND unassigned_node_count >= 0",
            name="ck_brain_profiles_counts",
        ),
        CheckConstraint(
            "relations_duration_ms >= 0 AND clustering_duration_ms >= 0 "
            "AND umap_duration_ms >= 0 AND labeling_duration_ms >= 0 "
            "AND total_duration_ms >= 0",
            name="ck_brain_profiles_durations",
        ),
        CheckConstraint(
            "label_strategy IN ('deterministic', 'ollama', 'mixed')",
            name="ck_brain_profiles_label_strategy",
        ),
        CheckConstraint(
            "status != 'ready' OR (completed_at IS NOT NULL AND activated_at IS NOT NULL)",
            name="ck_brain_profiles_ready_dates",
        ),
        UniqueConstraint(
            "logical_generation",
            name="uq_brain_profiles_logical_generation",
        ),
        Index(
            "uq_brain_profiles_single_ready",
            "status",
            unique=True,
            sqlite_where=text("status = 'ready'"),
        ),
        Index(
            "uq_brain_profiles_single_building",
            "status",
            unique=True,
            sqlite_where=text("status = 'building'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    embedding_profile_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("embedding_profiles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_model_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_semantic_text_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_logical_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    parameters_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="{}",
        server_default=text("'{}'"),
    )
    parameters_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    logical_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[BrainProfileStatus] = mapped_column(
        SqlEnum(
            BrainProfileStatus,
            name="brain_profile_status",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=BrainProfileStatus.BUILDING,
        server_default=text("'building'"),
    )
    knowledge_node_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    cluster_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    edge_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    unassigned_node_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    statistics_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="{}",
        server_default=text("'{}'"),
    )
    relations_duration_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    clustering_duration_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    umap_duration_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    labeling_duration_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    total_duration_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    label_strategy: Mapped[BrainLabelStrategy] = mapped_column(
        SqlEnum(
            BrainLabelStrategy,
            name="brain_label_strategy",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=BrainLabelStrategy.DETERMINISTIC,
        server_default=text("'deterministic'"),
    )
    label_model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    label_model_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    labels_generated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    embedding_profile: Mapped[EmbeddingProfile | None] = relationship()
    processing_jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="brain_profile",
        passive_deletes=True,
    )
    clusters: Mapped[list[BrainCluster]] = relationship(
        back_populates="brain_profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    node_layouts: Mapped[list[BrainNodeLayout]] = relationship(
        back_populates="brain_profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    edges: Mapped[list[BrainEdge]] = relationship(
        back_populates="brain_profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class BrainCluster(Base):
    __tablename__ = "brain_clusters"
    __table_args__ = (
        ForeignKeyConstraint(
            ["brain_profile_id", "parent_cluster_id"],
            ["brain_clusters.brain_profile_id", "brain_clusters.id"],
            name="fk_brain_clusters_parent_same_profile",
            ondelete="CASCADE",
        ),
        CheckConstraint("level >= 0", name="ck_brain_clusters_level"),
        CheckConstraint("length(trim(label)) > 0", name="ck_brain_clusters_label"),
        CheckConstraint(
            "label_source IN ('deterministic', 'ollama')",
            name="ck_brain_clusters_label_source",
        ),
        CheckConstraint("member_count > 0", name="ck_brain_clusters_member_count"),
        CheckConstraint(
            "x >= -1.0 AND x <= 1.0 AND y >= -1.0 AND y <= 1.0",
            name="ck_brain_clusters_coordinates",
        ),
        UniqueConstraint(
            "brain_profile_id",
            "id",
            name="uq_brain_clusters_profile_id",
        ),
        Index(
            "ix_brain_clusters_profile_level",
            "brain_profile_id",
            "level",
        ),
        Index(
            "ix_brain_clusters_profile_parent",
            "brain_profile_id",
            "parent_cluster_id",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    brain_profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("brain_profiles.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    parent_cluster_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    label_source: Mapped[BrainLabelSource] = mapped_column(
        SqlEnum(
            BrainLabelSource,
            name="brain_label_source",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=BrainLabelSource.DETERMINISTIC,
        server_default=text("'deterministic'"),
    )
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    centroid_json: Mapped[str] = mapped_column(Text, nullable=False)
    representative_nodes_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
        server_default=text("'[]'"),
    )
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    brain_profile: Mapped[BrainProfile] = relationship(
        back_populates="clusters",
        foreign_keys=[brain_profile_id],
        overlaps="children,parent",
    )
    parent: Mapped[BrainCluster | None] = relationship(
        back_populates="children",
        remote_side=lambda: [BrainCluster.brain_profile_id, BrainCluster.id],
        foreign_keys=lambda: [BrainCluster.brain_profile_id, BrainCluster.parent_cluster_id],
        overlaps="brain_profile,clusters",
    )
    children: Mapped[list[BrainCluster]] = relationship(
        back_populates="parent",
        foreign_keys=lambda: [BrainCluster.brain_profile_id, BrainCluster.parent_cluster_id],
        cascade="all",
        passive_deletes=True,
        overlaps="brain_profile,clusters",
    )
    node_layouts: Mapped[list[BrainNodeLayout]] = relationship(
        back_populates="cluster",
        passive_deletes=True,
        overlaps="brain_profile,node_layouts",
    )


class BrainNodeLayout(Base):
    __tablename__ = "brain_node_layouts"
    __table_args__ = (
        ForeignKeyConstraint(
            ["brain_profile_id", "cluster_id"],
            ["brain_clusters.brain_profile_id", "brain_clusters.id"],
            name="fk_brain_node_layouts_cluster_same_profile",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "x >= -1.0 AND x <= 1.0 AND y >= -1.0 AND y <= 1.0",
            name="ck_brain_node_layouts_coordinates",
        ),
        CheckConstraint(
            "(is_unassigned = 1 AND cluster_id IS NULL) "
            "OR (is_unassigned = 0 AND cluster_id IS NOT NULL)",
            name="ck_brain_node_layouts_assignment",
        ),
        CheckConstraint(
            "is_unassigned IN (0, 1)",
            name="ck_brain_node_layouts_is_unassigned",
        ),
        CheckConstraint(
            "membership_confidence IS NULL "
            "OR (membership_confidence >= 0.0 AND membership_confidence <= 1.0)",
            name="ck_brain_node_layouts_confidence",
        ),
        CheckConstraint(
            "representative_rank IS NULL OR representative_rank > 0",
            name="ck_brain_node_layouts_representative_rank",
        ),
        Index(
            "ix_brain_node_layouts_profile_cluster",
            "brain_profile_id",
            "cluster_id",
        ),
    )

    brain_profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("brain_profiles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    knowledge_node_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    cluster_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    is_unassigned: Mapped[bool] = mapped_column(
        Boolean(),
        nullable=False,
        default=False,
        server_default=text("0"),
    )
    membership_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    representative_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    brain_profile: Mapped[BrainProfile] = relationship(
        back_populates="node_layouts",
        foreign_keys=[brain_profile_id],
        overlaps="cluster,node_layouts",
    )
    knowledge_node: Mapped[KnowledgeNode] = relationship()
    cluster: Mapped[BrainCluster | None] = relationship(
        back_populates="node_layouts",
        foreign_keys=[brain_profile_id, cluster_id],
        overlaps="brain_profile,node_layouts",
    )


class BrainEdge(Base):
    __tablename__ = "brain_edges"
    __table_args__ = (
        ForeignKeyConstraint(
            ["brain_profile_id", "source_node_id"],
            ["brain_node_layouts.brain_profile_id", "brain_node_layouts.knowledge_node_id"],
            name="fk_brain_edges_source_layout",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["brain_profile_id", "target_node_id"],
            ["brain_node_layouts.brain_profile_id", "brain_node_layouts.knowledge_node_id"],
            name="fk_brain_edges_target_layout",
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "source_node_id != target_node_id",
            name="ck_brain_edges_distinct_nodes",
        ),
        CheckConstraint(
            "source_node_id < target_node_id",
            name="ck_brain_edges_canonical_order",
        ),
        CheckConstraint(
            "cosine_score >= -1.0 AND cosine_score <= 1.0",
            name="ck_brain_edges_cosine_score",
        ),
        CheckConstraint("tag_bonus >= 0.0", name="ck_brain_edges_tag_bonus"),
        CheckConstraint(
            "source_rank IS NULL OR source_rank > 0",
            name="ck_brain_edges_source_rank",
        ),
        CheckConstraint(
            "target_rank IS NULL OR target_rank > 0",
            name="ck_brain_edges_target_rank",
        ),
        CheckConstraint(
            "is_mutual IN (0, 1)",
            name="ck_brain_edges_is_mutual",
        ),
        UniqueConstraint(
            "brain_profile_id",
            "source_node_id",
            "target_node_id",
            name="uq_brain_edges_profile_nodes",
        ),
        Index(
            "ix_brain_edges_profile_score",
            "brain_profile_id",
            "final_score",
        ),
        Index(
            "ix_brain_edges_profile_source",
            "brain_profile_id",
            "source_node_id",
        ),
        Index(
            "ix_brain_edges_profile_target",
            "brain_profile_id",
            "target_node_id",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    brain_profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("brain_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_node_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    target_node_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    cosine_score: Mapped[float] = mapped_column(Float, nullable=False)
    tag_bonus: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default=text("0"),
    )
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_mutual: Mapped[bool] = mapped_column(
        Boolean(),
        nullable=False,
        default=False,
        server_default=text("0"),
    )
    source_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    brain_profile: Mapped[BrainProfile] = relationship(
        back_populates="edges",
        foreign_keys=[brain_profile_id],
    )
