from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from second_brain.db.models.processing import ProcessingJobStatus

BrainState = Literal[
    "empty",
    "not_built",
    "building",
    "ready",
    "stale",
    "error",
    "vector_index_required",
    "unavailable",
]


class BrainSimilarityStats(BaseModel):
    minimum: float | None = None
    mean: float | None = None
    median: float | None = None
    maximum: float | None = None


class BrainClusterSizeStats(BaseModel):
    minimum: int | None = None
    mean: float | None = None
    maximum: int | None = None


class BrainProfileOut(BaseModel):
    id: UUID
    logical_generation: int
    status: Literal["building", "ready", "stale", "error"]
    embedding_profile_id: UUID | None
    embedding_provider: str
    embedding_model_name: str
    embedding_model_digest: str | None
    embedding_dimensions: int
    embedding_semantic_text_version: str
    embedding_logical_generation: int
    algorithm_version: str
    knowledge_node_count: int
    cluster_count: int
    edge_count: int
    unassigned_node_count: int
    cluster_counts_by_level: dict[str, int] = Field(default_factory=dict)
    similarity: BrainSimilarityStats = Field(default_factory=BrainSimilarityStats)
    cluster_sizes: BrainClusterSizeStats = Field(default_factory=BrainClusterSizeStats)
    relations_duration_ms: int
    clustering_duration_ms: int
    umap_duration_ms: int
    labeling_duration_ms: int
    total_duration_ms: int
    label_strategy: Literal["deterministic", "ollama", "mixed"]
    label_model_name: str | None
    label_model_digest: str | None
    created_at: datetime
    completed_at: datetime | None
    activated_at: datetime | None
    error_message: str | None

    @field_validator("created_at", "completed_at", "activated_at")
    @classmethod
    def force_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class BrainJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    brain_profile_id: UUID | None
    kind: Literal["build_brain", "relabel_brain"]
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
    attempt_count: int
    llm_call_count: int
    llm_retry_count: int
    llm_duration_ms: int
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
    def force_job_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class BrainStatusOut(BaseModel):
    state: BrainState
    active_profile: BrainProfileOut | None = None
    building_profile: BrainProfileOut | None = None
    active_job: BrainJobOut | None = None
    latest_job: BrainJobOut | None = None
    stale_reasons: list[str] = Field(default_factory=list)
    can_rebuild: bool
    can_relabel: bool
    error: str | None = None


class BrainRebuildRequest(BaseModel):
    confirm: Literal[True]


class BrainRelabelRequest(BaseModel):
    confirm: Literal[True]


class BrainClusterOut(BaseModel):
    id: UUID
    parent_id: UUID | None
    level: int
    label: str
    description: str | None
    label_source: Literal["deterministic", "ollama"]
    member_count: int
    representative_knowledge_node_ids: list[UUID]
    x: float
    y: float
    child_count: int = 0


class BrainKnowledgeNodeOut(BaseModel):
    id: UUID
    cluster_id: UUID | None
    title: str
    tags: list[str]
    source_id: UUID
    source_title: str
    x: float
    y: float
    is_unassigned: bool
    href: str


class BrainClusterDetail(BrainClusterOut):
    children: list[BrainClusterOut]
    knowledge_nodes: list[BrainKnowledgeNodeOut]


class BrainGraphNode(BaseModel):
    id: str
    kind: Literal["cluster", "knowledge"]
    label: str
    x: float
    y: float
    size: int
    cluster_id: UUID | None = None
    knowledge_node_id: UUID | None = None
    source_id: UUID | None = None
    tags: list[str] = Field(default_factory=list)
    href: str | None = None


class BrainGraphEdge(BaseModel):
    source: str
    target: str
    score: float
    relation_count: int = 1


class BrainGraphOut(BaseModel):
    profile_id: UUID
    level: int
    parent_cluster_id: UUID | None
    nodes: list[BrainGraphNode]
    edges: list[BrainGraphEdge]
    truncated: bool = False


class BrainSearchAncestor(BaseModel):
    id: UUID
    label: str
    level: int


class BrainSearchResult(BaseModel):
    kind: Literal["cluster", "knowledge"]
    target_id: UUID
    label: str
    level: int | None
    cluster_id: UUID | None
    x: float
    y: float
    member_count: int | None = None
    tags: list[str] = Field(default_factory=list)
    source_id: UUID | None = None
    source_title: str | None = None
    href: str | None = None
    ancestors: list[BrainSearchAncestor] = Field(default_factory=list)


class BrainSearchOut(BaseModel):
    profile_id: UUID
    query: str
    items: list[BrainSearchResult]
