from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from second_brain.db.models.processing import ProcessingJobKind, ProcessingJobStatus


class AnalysisJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_id: UUID
    kind: ProcessingJobKind
    status: ProcessingJobStatus
    stage: str | None
    progress_current: int
    progress_total: int
    progress_percent: int
    progress_message: str | None
    error_message: str | None
    error_code: str | None
    error_type: str | None
    error_detail: str | None
    error_stage: str | None
    error_passage_id: UUID | None
    error_passage_index: int | None
    error_attempt: int | None
    error_call_type: str | None
    attempt_count: int
    llm_call_count: int
    llm_retry_count: int
    llm_duration_ms: int
    ollama_total_duration_ns: int
    prompt_eval_count: int
    prompt_eval_duration_ns: int
    eval_count: int
    eval_duration_ns: int
    knowledge_node_count: int
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    heartbeat_at: datetime
    is_stale: bool = False
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator(
        "created_at",
        "updated_at",
        "last_activity_at",
        "heartbeat_at",
        "started_at",
        "finished_at",
    )
    @classmethod
    def force_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
