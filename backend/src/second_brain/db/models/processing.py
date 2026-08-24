from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    text,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from second_brain.db.base import Base, UTCDateTime, utc_now

if TYPE_CHECKING:
    from second_brain.db.models.embedding import EmbeddingProfile
    from second_brain.db.models.source import Source


class ProcessingJobKind(str, Enum):
    ANALYZE_SOURCE = "analyze_source"
    INDEX_KNOWLEDGE = "index_knowledge"
    REBUILD_VECTOR_INDEX = "rebuild_vector_index"


class ProcessingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('analyze_source', 'index_knowledge', 'rebuild_vector_index')",
            name="processing_job_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed')",
            name="processing_job_status",
        ),
        CheckConstraint(
            "kind != 'analyze_source' OR source_id IS NOT NULL",
            name="ck_processing_jobs_analysis_source",
        ),
        CheckConstraint(
            "progress_current >= 0",
            name="ck_processing_jobs_progress_current",
        ),
        CheckConstraint(
            "progress_total >= 0",
            name="ck_processing_jobs_progress_total",
        ),
        CheckConstraint(
            "progress_percent >= 0 AND progress_percent <= 100",
            name="ck_processing_jobs_progress_percent",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_processing_jobs_attempt_count",
        ),
        CheckConstraint(
            "error_passage_index IS NULL OR error_passage_index >= 0",
            name="ck_processing_jobs_error_passage_index",
        ),
        CheckConstraint(
            "error_attempt IS NULL OR error_attempt >= 0",
            name="ck_processing_jobs_error_attempt",
        ),
        CheckConstraint(
            "llm_call_count >= 0 AND llm_retry_count >= 0 "
            "AND llm_duration_ms >= 0 AND ollama_total_duration_ns >= 0 "
            "AND prompt_eval_count >= 0 AND prompt_eval_duration_ns >= 0 "
            "AND eval_count >= 0 AND eval_duration_ns >= 0 AND knowledge_node_count >= 0",
            name="ck_processing_jobs_analysis_metrics",
        ),
        CheckConstraint(
            "embedding_batch_count >= 0 AND embedding_item_count >= 0 "
            "AND embedding_duration_ms >= 0 AND embedding_total_duration_ns >= 0 "
            "AND embedding_prompt_eval_count >= 0",
            name="ck_processing_jobs_embedding_metrics",
        ),
        Index("ix_processing_jobs_status_created_at", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    source_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    embedding_profile_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("embedding_profiles.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    kind: Mapped[ProcessingJobKind] = mapped_column(
        SqlEnum(
            ProcessingJobKind,
            name="processing_job_kind",
            native_enum=False,
            create_constraint=False,
            length=20,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=ProcessingJobKind.ANALYZE_SOURCE,
        server_default=text("'analyze_source'"),
    )
    status: Mapped[ProcessingJobStatus] = mapped_column(
        SqlEnum(
            ProcessingJobStatus,
            name="processing_job_status",
            native_enum=False,
            create_constraint=False,
            values_callable=lambda members: [member.value for member in members],
        ),
        nullable=False,
        default=ProcessingJobStatus.PENDING,
        server_default=text("'pending'"),
    )
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress_current: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    progress_total: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    progress_percent: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    progress_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_passage_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("source_passages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    error_passage_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_attempt: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_call_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    llm_call_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    llm_retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    llm_duration_ms: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    ollama_total_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    prompt_eval_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    prompt_eval_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    eval_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    eval_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    knowledge_node_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    embedding_batch_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    embedding_item_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    embedding_duration_ms: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    embedding_total_duration_ns: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
    embedding_prompt_eval_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default=text("0")
    )
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
    last_activity_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)

    source: Mapped[Source | None] = relationship(back_populates="processing_jobs")
    embedding_profile: Mapped[EmbeddingProfile | None] = relationship(
        back_populates="processing_jobs"
    )

    @property
    def heartbeat_at(self) -> datetime:
        return self.last_activity_at


def processing_job_is_stale(
    job: ProcessingJob,
    *,
    stale_after_seconds: int,
    now: datetime | None = None,
) -> bool:
    if job.status != ProcessingJobStatus.RUNNING:
        return False
    reference_time = now or utc_now()
    return job.last_activity_at <= reference_time - timedelta(seconds=stale_after_seconds)
