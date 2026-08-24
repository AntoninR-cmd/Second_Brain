from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from second_brain.rag.answer_schema import RagMode
from second_brain.schemas.knowledge import (
    KnowledgeEvidenceOut,
    KnowledgeNodeSummary,
    KnowledgeSourceOut,
)


class RagQuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2_000)
    mode: RagMode = "brain_only"
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized) < 2:
            raise ValueError("La question doit contenir au moins deux caractères.")
        return normalized


class RagKnowledgeOut(BaseModel):
    context_id: str | None
    score: float
    href: str
    provided_to_model: bool
    used: bool
    knowledge_node: KnowledgeNodeSummary
    source: KnowledgeSourceOut
    evidences: list[KnowledgeEvidenceOut]


class RagTimingsOut(BaseModel):
    readiness_ms: float = Field(ge=0)
    embedding_ms: float = Field(ge=0)
    qdrant_ms: float = Field(ge=0)
    retrieval_sqlite_ms: float = Field(ge=0)
    context_build_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    provenance_validation_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)


class RagAnswerResponse(BaseModel):
    request_id: UUID
    question: str
    mode: RagMode
    answer: str
    model_additions: str | None
    insufficient_context: bool
    generation_model: str
    retrieved_knowledge: list[RagKnowledgeOut]
    used_knowledge: list[RagKnowledgeOut]
    timings: RagTimingsOut
    citation_format: Literal["[Kx]"] = "[Kx]"
