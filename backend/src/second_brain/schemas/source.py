from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from second_brain.db.models.source import AnalysisStatus, ProcessingStatus, SourceType


class ManualSourceCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    author: str | None = Field(default=None, max_length=255)
    text: str = Field(min_length=1)

    @field_validator("title", "author")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Le texte ne peut pas être vide.")
        return value


class SourceSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    type: SourceType
    title: str
    author: str | None
    original_filename: str | None
    processing_status: ProcessingStatus
    analysis_status: AnalysisStatus
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def force_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class SourceDetail(SourceSummary):
    original_file_path: str | None
    file_sha256: str | None
    raw_text: str
    segment_count: int = Field(default=0, ge=0)
    summary: str | None
    analysis_error: str | None
    analysis_started_at: datetime | None
    analysis_completed_at: datetime | None
    knowledge_count: int = Field(default=0, ge=0)

    @field_validator("analysis_started_at", "analysis_completed_at")
    @classmethod
    def force_optional_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class SourceList(BaseModel):
    items: list[SourceSummary]
    next_cursor: UUID | None


class SourceSegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_id: UUID
    index: int
    text: str
    start_ms: int | None
    end_ms: int | None


class SourceSegmentList(BaseModel):
    items: list[SourceSegmentOut]
    next_cursor: int | None
