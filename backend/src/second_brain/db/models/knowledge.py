from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from second_brain.db.base import Base, UTCDateTime, utc_now

if TYPE_CHECKING:
    from second_brain.db.models.source import Source
    from second_brain.db.models.source_passage import SourcePassage
    from second_brain.db.models.source_segment import SourceSegment
    from second_brain.db.models.taxonomy import KnowledgeNodeTag


class KnowledgeNode(Base):
    __tablename__ = "knowledge_nodes"

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    source_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    source: Mapped[Source] = relationship(back_populates="knowledge_nodes")
    evidence: Mapped[list[KnowledgeEvidence]] = relationship(
        back_populates="knowledge_node",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="KnowledgeEvidence.evidence_index",
    )
    tag_links: Mapped[list[KnowledgeNodeTag]] = relationship(
        back_populates="knowledge_node",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class KnowledgeEvidence(Base):
    __tablename__ = "knowledge_evidence"
    __table_args__ = (
        CheckConstraint(
            "start_ms IS NULL OR end_ms IS NULL OR start_ms <= end_ms",
            name="ck_knowledge_evidence_time_range",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_end IS NULL OR char_start <= char_end",
            name="ck_knowledge_evidence_char_range",
        ),
        UniqueConstraint(
            "knowledge_node_id",
            "evidence_index",
            name="uq_knowledge_evidence_node_index",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    knowledge_node_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    passage_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("source_passages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    evidence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    first_segment_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("source_segments.id", ondelete="CASCADE"),
        nullable=True,
    )
    last_segment_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("source_segments.id", ondelete="CASCADE"),
        nullable=True,
    )
    original_excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    start_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    end_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    knowledge_node: Mapped[KnowledgeNode] = relationship(back_populates="evidence")
    passage: Mapped[SourcePassage] = relationship(back_populates="evidence")
    first_segment: Mapped[SourceSegment | None] = relationship(
        foreign_keys=[first_segment_id],
    )
    last_segment: Mapped[SourceSegment | None] = relationship(
        foreign_keys=[last_segment_id],
    )
