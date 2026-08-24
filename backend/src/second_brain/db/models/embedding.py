from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
    inspect,
    text,
    update,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, mapped_column, relationship

from second_brain.db.base import Base, UTCDateTime, utc_now

if TYPE_CHECKING:
    from second_brain.db.models.knowledge import KnowledgeNode
    from second_brain.db.models.processing import ProcessingJob


class EmbeddingProfileStatus(str, Enum):
    BUILDING = "building"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


class EmbeddingDistance(str, Enum):
    COSINE = "cosine"


class KnowledgeEmbeddingStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    STALE = "stale"
    FAILED = "failed"


class EmbeddingProfile(Base):
    """Versioned metadata for one homogeneous vector space."""

    __tablename__ = "embedding_profiles"
    __table_args__ = (
        CheckConstraint("length(trim(provider)) > 0", name="ck_embedding_profiles_provider"),
        CheckConstraint("length(trim(model_name)) > 0", name="ck_embedding_profiles_model"),
        CheckConstraint(
            "dimensions IS NULL OR dimensions > 0",
            name="ck_embedding_profiles_dimensions",
        ),
        CheckConstraint(
            "logical_generation > 0",
            name="ck_embedding_profiles_logical_generation",
        ),
        CheckConstraint(
            "distance IN ('cosine')",
            name="ck_embedding_profiles_distance",
        ),
        CheckConstraint(
            "status IN ('building', 'active', 'retired', 'failed')",
            name="ck_embedding_profiles_status",
        ),
        CheckConstraint(
            "status != 'active' OR (dimensions IS NOT NULL AND activated_at IS NOT NULL)",
            name="ck_embedding_profiles_active_metadata",
        ),
        UniqueConstraint(
            "logical_generation",
            name="uq_embedding_profiles_logical_generation",
        ),
        UniqueConstraint(
            "collection_name",
            name="uq_embedding_profiles_collection_name",
        ),
        Index(
            "uq_embedding_profiles_single_active",
            "status",
            unique=True,
            sqlite_where=text("status = 'active'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="ollama",
        server_default=text("'ollama'"),
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_digest: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distance: Mapped[EmbeddingDistance] = mapped_column(
        SqlEnum(
            EmbeddingDistance,
            name="embedding_distance",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=EmbeddingDistance.COSINE,
        server_default=text("'cosine'"),
    )
    collection_name: Mapped[str] = mapped_column(String(255), nullable=False)
    semantic_text_version: Mapped[str] = mapped_column(String(64), nullable=False)
    logical_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[EmbeddingProfileStatus] = mapped_column(
        SqlEnum(
            EmbeddingProfileStatus,
            name="embedding_profile_status",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=EmbeddingProfileStatus.BUILDING,
        server_default=text("'building'"),
    )
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
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    knowledge_embeddings: Mapped[list[KnowledgeEmbedding]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    processing_jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="embedding_profile",
        passive_deletes=True,
    )


class KnowledgeEmbedding(Base):
    """SQLite checkpoint for one node in one embedding profile."""

    __tablename__ = "knowledge_embeddings"
    __table_args__ = (
        CheckConstraint(
            "length(text_fingerprint) = 64",
            name="ck_knowledge_embeddings_fingerprint",
        ),
        CheckConstraint(
            "status IN ('pending', 'indexed', 'stale', 'failed')",
            name="ck_knowledge_embeddings_status",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_knowledge_embeddings_attempt_count",
        ),
        CheckConstraint(
            "status != 'indexed' OR indexed_at IS NOT NULL",
            name="ck_knowledge_embeddings_indexed_at",
        ),
        Index(
            "ix_knowledge_embeddings_profile_status",
            "embedding_profile_id",
            "status",
        ),
    )

    knowledge_node_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding_profile_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("embedding_profiles.id", ondelete="CASCADE"),
        primary_key=True,
    )
    text_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[KnowledgeEmbeddingStatus] = mapped_column(
        SqlEnum(
            KnowledgeEmbeddingStatus,
            name="knowledge_embedding_status",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=KnowledgeEmbeddingStatus.PENDING,
        server_default=text("'pending'"),
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
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
    indexed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    knowledge_node: Mapped[KnowledgeNode] = relationship(back_populates="embeddings")
    profile: Mapped[EmbeddingProfile] = relationship(back_populates="knowledge_embeddings")


from second_brain.db.models.knowledge import KnowledgeNode  # noqa: E402


@event.listens_for(KnowledgeNode, "after_update")
def mark_node_embeddings_stale(
    mapper: Mapper[KnowledgeNode],
    connection: Connection,
    target: KnowledgeNode,
) -> None:
    """Persist staleness when semantic title/content changes through the ORM."""

    del mapper
    state = inspect(target)
    if not (state.attrs.title.history.has_changes() or state.attrs.content.history.has_changes()):
        return
    connection.execute(
        update(KnowledgeEmbedding)
        .where(KnowledgeEmbedding.knowledge_node_id == target.id)
        .values(
            status=KnowledgeEmbeddingStatus.STALE,
            updated_at=utc_now(),
        )
    )
