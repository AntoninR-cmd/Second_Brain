from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import (
    text as sql_text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from second_brain.db.base import Base, UTCDateTime, utc_now

if TYPE_CHECKING:
    from second_brain.db.models.knowledge import KnowledgeEvidence
    from second_brain.db.models.source import Source
    from second_brain.db.models.source_segment import SourceSegment


class SourcePassageAnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SourcePassage(Base):
    """A bounded working passage; faithful source segments remain unchanged."""

    __tablename__ = "source_passages"
    __table_args__ = (
        CheckConstraint("token_count >= 0", name="ck_source_passages_token_count"),
        CheckConstraint(
            "first_segment_index IS NULL OR last_segment_index IS NULL "
            "OR first_segment_index <= last_segment_index",
            name="ck_source_passages_segment_range",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_end IS NULL OR char_start <= char_end",
            name="ck_source_passages_char_range",
        ),
        CheckConstraint(
            "analysis_status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_source_passages_analysis_status",
        ),
        CheckConstraint(
            "analysis_attempt_count >= 0",
            name="ck_source_passages_analysis_attempt_count",
        ),
        CheckConstraint(
            "llm_call_count >= 0 AND llm_retry_count >= 0 "
            "AND llm_duration_ms >= 0 AND ollama_total_duration_ns >= 0 "
            "AND prompt_eval_count >= 0 AND prompt_eval_duration_ns >= 0 "
            "AND eval_count >= 0 AND eval_duration_ns >= 0 AND knowledge_count >= 0",
            name="ck_source_passages_analysis_metrics",
        ),
        CheckConstraint(
            "analysis_status != 'completed' OR "
            "(analysis_payload_json IS NOT NULL AND intermediate_summary IS NOT NULL)",
            name="ck_source_passages_completed_payload",
        ),
        UniqueConstraint(
            "source_id",
            "passage_index",
            name="uq_source_passages_source_id_index",
        ),
    )

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
    index: Mapped[int] = mapped_column("passage_index", Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    first_segment_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_segment_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    intermediate_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_status: Mapped[SourcePassageAnalysisStatus] = mapped_column(
        SqlEnum(
            SourcePassageAnalysisStatus,
            name="source_passage_analysis_status",
            native_enum=False,
            create_constraint=False,
            length=16,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=SourcePassageAnalysisStatus.PENDING,
        server_default=sql_text("'pending'"),
    )
    analysis_payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=sql_text("0"),
    )
    analysis_started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    analysis_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    analysis_last_activity_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
        default=utc_now,
    )
    llm_call_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=sql_text("0")
    )
    llm_retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=sql_text("0")
    )
    llm_duration_ms: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    ollama_total_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    prompt_eval_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    prompt_eval_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    eval_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    eval_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=sql_text("0")
    )
    knowledge_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=sql_text("0")
    )

    source: Mapped[Source] = relationship(back_populates="passages")
    segment_links: Mapped[list[SourcePassageSegment]] = relationship(
        back_populates="passage",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SourcePassageSegment.position",
    )
    evidence: Mapped[list[KnowledgeEvidence]] = relationship(
        back_populates="passage",
        passive_deletes=True,
    )


class SourcePassageSegment(Base):
    """Ordered, many-to-many provenance between passages and exact segments."""

    __tablename__ = "source_passage_segments"
    __table_args__ = (
        CheckConstraint("position >= 0", name="ck_source_passage_segments_position"),
        UniqueConstraint(
            "passage_id",
            "position",
            name="uq_source_passage_segments_passage_position",
        ),
    )

    passage_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("source_passages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    segment_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("source_segments.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    passage: Mapped[SourcePassage] = relationship(back_populates="segment_links")
    segment: Mapped[SourceSegment] = relationship(back_populates="passage_links")
