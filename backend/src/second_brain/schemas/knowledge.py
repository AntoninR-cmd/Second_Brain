from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, field_validator

from second_brain.db.models.source import SourceType


class KnowledgeNodeSummary(BaseModel):
    id: UUID
    source_id: UUID
    title: str
    content: str
    tags: list[str]
    evidence_count: int
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def force_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class KnowledgeNodeList(BaseModel):
    items: list[KnowledgeNodeSummary]
    next_cursor: UUID | None


class KnowledgeSourceOut(BaseModel):
    id: UUID
    title: str
    type: SourceType
    author: str | None
    original_filename: str | None
    original_file_path: str | None


class KnowledgeEvidenceOut(BaseModel):
    id: UUID
    passage_id: UUID
    passage_index: int
    original_excerpt: str
    start_ms: int | None
    end_ms: int | None
    first_segment_index: int | None
    last_segment_index: int | None
    char_start: int | None
    char_end: int | None


class KnowledgeNodeDetail(KnowledgeNodeSummary):
    source: KnowledgeSourceOut
    evidences: list[KnowledgeEvidenceOut]
