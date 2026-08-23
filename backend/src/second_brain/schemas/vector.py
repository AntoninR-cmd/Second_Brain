from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from second_brain.db.models.processing import ProcessingJobStatus
from second_brain.schemas.knowledge import (
    KnowledgeEvidenceOut,
    KnowledgeNodeSummary,
    KnowledgeSourceOut,
)

VectorIndexState = Literal[
    "empty",
    "not_built",
    "building",
    "ready",
    "stale",
    "incompatible",
    "unavailable",
    "corrupt",
]


class EmbeddingReadinessOut(BaseModel):
    ollama_available: bool
    configured_model: str
    model_available: bool
    error: str | None = None


class EmbeddingProfileOut(BaseModel):
    id: UUID
    model_name: str
    dimensions: int | None
    distance: Literal["cosine"] = "cosine"
    logical_generation: int


class VectorJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: Literal["index_knowledge", "rebuild_vector_index"]
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
    attempt_count: int
    embedding_batch_count: int
    embedding_item_count: int
    embedding_duration_ms: int
    embedding_total_duration_ns: int
    embedding_prompt_eval_count: int
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    is_stale: bool = False
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator(
        "created_at",
        "updated_at",
        "last_activity_at",
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


class VectorIndexStatusOut(BaseModel):
    state: VectorIndexState
    configured_model: str
    embedding: EmbeddingReadinessOut
    total_nodes: int
    indexed_nodes: int
    pending_or_stale_nodes: int
    failed_nodes: int
    orphan_points: int = 0
    active_profile: EmbeddingProfileOut | None = None
    active_job: VectorJobOut | None = None
    error: str | None = None


class RebuildVectorIndexRequest(BaseModel):
    confirm: Literal[True]


class SemanticSearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 2:
            raise ValueError("La recherche doit contenir au moins deux caracteres.")
        return normalized


class SemanticSearchItem(BaseModel):
    score: float
    href: str
    knowledge_node: KnowledgeNodeSummary
    source: KnowledgeSourceOut
    evidences: list[KnowledgeEvidenceOut]


class SemanticSearchProfileOut(BaseModel):
    model_name: str
    dimensions: int
    distance: Literal["cosine"] = "cosine"


class SemanticSearchResponse(BaseModel):
    query: str
    items: list[SemanticSearchItem]
    profile: SemanticSearchProfileOut | None
