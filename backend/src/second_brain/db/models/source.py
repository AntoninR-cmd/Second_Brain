from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, String, Text, Uuid, text
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from second_brain.db.base import Base, UTCDateTime, utc_now

if TYPE_CHECKING:
    from second_brain.db.models.knowledge import KnowledgeNode
    from second_brain.db.models.processing import ProcessingJob
    from second_brain.db.models.source_passage import SourcePassage
    from second_brain.db.models.source_segment import SourceSegment


class SourceType(str, Enum):
    MANUAL = "manual"
    SRT = "srt"
    TXT = "txt"


class ProcessingStatus(str, Enum):
    READY = "ready"


class AnalysisStatus(str, Enum):
    NOT_ANALYZED = "not_analyzed"
    QUEUED = "queued"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    ERROR = "error"


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        CheckConstraint(
            "type IN ('manual', 'srt', 'txt')",
            name="source_type",
        ),
        CheckConstraint(
            "processing_status IN ('ready')",
            name="processing_status",
        ),
        CheckConstraint(
            "analysis_status IN ('not_analyzed', 'queued', 'processing', 'analyzed', 'error')",
            name="ck_sources_analysis_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    type: Mapped[SourceType] = mapped_column(
        SqlEnum(
            SourceType,
            name="source_type",
            native_enum=False,
            create_constraint=False,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=SourceType.MANUAL,
        server_default=text("'manual'"),
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    original_file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        SqlEnum(
            ProcessingStatus,
            name="processing_status",
            native_enum=False,
            create_constraint=False,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=ProcessingStatus.READY,
        server_default=text("'ready'"),
    )
    analysis_status: Mapped[AnalysisStatus] = mapped_column(
        SqlEnum(
            AnalysisStatus,
            name="analysis_status",
            native_enum=False,
            create_constraint=False,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=AnalysisStatus.NOT_ANALYZED,
        server_default=text("'not_analyzed'"),
        index=True,
    )
    analysis_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_started_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
    analysis_completed_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(),
        nullable=True,
    )
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
    segments: Mapped[list[SourceSegment]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SourceSegment.index",
    )
    passages: Mapped[list[SourcePassage]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SourcePassage.index",
    )
    knowledge_nodes: Mapped[list[KnowledgeNode]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="KnowledgeNode.created_at",
    )
    processing_jobs: Mapped[list[ProcessingJob]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProcessingJob.created_at",
    )
