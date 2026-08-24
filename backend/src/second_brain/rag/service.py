from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from second_brain.core.config import Settings
from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.db.models.taxonomy import KnowledgeNodeTag
from second_brain.db.session import Database
from second_brain.llm.client import GenerationAttemptMetrics, TextGenerator
from second_brain.llm.errors import StructuredOutputValidationError
from second_brain.llm.prompt_loader import build_rag_answer_prompt, rag_system_prompt
from second_brain.rag.answer_schema import BrainOnlyAnswer, BrainPlusModelAnswer, RagMode
from second_brain.rag.citation_validator import (
    INSUFFICIENT_CONTEXT_ANSWER,
    ValidatedRagAnswer,
    validate_rag_answer,
)
from second_brain.rag.context_builder import BuiltRagContext, build_rag_context
from second_brain.services.vector_index import SemanticSearchResults, VectorIndexService
from second_brain.vector.semantic_text import semantic_text_fingerprint

logger = logging.getLogger(__name__)


class RagError(RuntimeError):
    code = "rag_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RagInvalidAnswerError(RagError):
    code = "rag_invalid_answer"


class RagContextChangedError(RagError):
    code = "rag_context_changed"


@dataclass(frozen=True, slots=True)
class RagTimings:
    readiness_seconds: float
    embedding_seconds: float
    qdrant_seconds: float
    retrieval_sqlite_seconds: float
    context_build_seconds: float
    generation_seconds: float
    provenance_validation_seconds: float
    total_seconds: float
    prompt_eval_count: int | None = None
    eval_count: int | None = None


@dataclass(frozen=True, slots=True)
class RagKnowledgeResult:
    context_id: str | None
    score: float
    node: KnowledgeNode
    provided_to_model: bool
    used: bool


@dataclass(frozen=True, slots=True)
class RagAnswerResult:
    request_id: UUID
    question: str
    mode: RagMode
    answer: str
    model_additions: str | None
    insufficient_context: bool
    retrieved_knowledge: tuple[RagKnowledgeResult, ...]
    used_knowledge: tuple[RagKnowledgeResult, ...]
    timings: RagTimings
    generation_model: str


class RagService:
    """Answer one independent question from the reconstructible semantic index."""

    def __init__(
        self,
        *,
        database: Database,
        vector_service: VectorIndexService,
        generator: TextGenerator,
        settings: Settings,
        work_lock: asyncio.Lock,
    ) -> None:
        self._database = database
        self._vector_service = vector_service
        self._generator = generator
        self._settings = settings
        self._work_lock = work_lock

    async def answer(
        self,
        question: str,
        *,
        mode: RagMode = "brain_only",
        top_k: int | None = None,
    ) -> RagAnswerResult:
        request_id = uuid4()
        started_at = perf_counter()
        selected_top_k = top_k or self._settings.rag_retrieval_top_k
        try:
            async with self._work_lock:
                retrieval = await self._vector_service.search(question, top_k=selected_top_k)
                context_started_at = perf_counter()
                context = self._build_context(retrieval)
                context_seconds = perf_counter() - context_started_at

                generation_metrics: list[GenerationAttemptMetrics] = []
                generation_seconds = 0.0
                if mode == "brain_only" and not context.entries:
                    validated = ValidatedRagAnswer(
                        brain_answer=INSUFFICIENT_CONTEXT_ANSWER,
                        model_additions=None,
                        used_knowledge=(),
                        insufficient_context=True,
                    )
                else:
                    generation_started_at = perf_counter()
                    validated = await self._generate_answer(
                        question=question,
                        mode=mode,
                        context=context,
                        metrics=generation_metrics,
                    )
                    generation_seconds = perf_counter() - generation_started_at

                provenance_started_at = perf_counter()
                fresh_nodes = await self._revalidate_retrieval(retrieval)
                provenance_seconds = perf_counter() - provenance_started_at

            knowledge_results = _build_knowledge_results(
                retrieval,
                context,
                fresh_nodes,
                used_references=frozenset(validated.used_knowledge),
            )
            used_by_reference = {item.context_id: item for item in knowledge_results if item.used}
            used_results = tuple(
                used_by_reference[reference] for reference in validated.used_knowledge
            )
            attempts = tuple(generation_metrics)
            timings = RagTimings(
                readiness_seconds=retrieval.timings.readiness_seconds,
                embedding_seconds=retrieval.timings.embedding_seconds,
                qdrant_seconds=retrieval.timings.qdrant_seconds,
                retrieval_sqlite_seconds=retrieval.timings.sqlite_seconds,
                context_build_seconds=context_seconds,
                generation_seconds=generation_seconds,
                provenance_validation_seconds=provenance_seconds,
                total_seconds=perf_counter() - started_at,
                prompt_eval_count=_sum_optional(metric.prompt_eval_count for metric in attempts),
                eval_count=_sum_optional(metric.eval_count for metric in attempts),
            )
            logger.info(
                "RAG answer completed request_id=%s mode=%s retrieved=%d provided=%d "
                "used=%d insufficient=%s embedding_seconds=%.3f qdrant_seconds=%.3f "
                "context_seconds=%.3f generation_seconds=%.3f total_seconds=%.3f "
                "prompt_eval_count=%s eval_count=%s",
                request_id,
                mode,
                len(retrieval.items),
                len(context.entries),
                len(used_results),
                validated.insufficient_context,
                timings.embedding_seconds,
                timings.qdrant_seconds,
                timings.context_build_seconds,
                timings.generation_seconds,
                timings.total_seconds,
                timings.prompt_eval_count,
                timings.eval_count,
            )
            return RagAnswerResult(
                request_id=request_id,
                question=question,
                mode=mode,
                answer=validated.brain_answer,
                model_additions=validated.model_additions,
                insufficient_context=validated.insufficient_context,
                retrieved_knowledge=knowledge_results,
                used_knowledge=used_results,
                timings=timings,
                generation_model=self._settings.ollama_generation_model,
            )
        except Exception as error:
            logger.warning(
                "RAG answer failed request_id=%s mode=%s error_type=%s total_seconds=%.3f",
                request_id,
                mode,
                type(error).__name__,
                perf_counter() - started_at,
            )
            raise

    def _build_context(self, retrieval: SemanticSearchResults) -> BuiltRagContext:
        return build_rag_context(
            retrieval.items,
            max_nodes=self._settings.rag_context_max_nodes,
            max_chars=self._settings.rag_context_max_chars,
            knowledge_max_chars=self._settings.rag_knowledge_max_chars,
            max_evidence_per_node=self._settings.rag_max_evidence_per_node,
            evidence_max_chars=self._settings.rag_evidence_max_chars,
        )

    async def _generate_answer(
        self,
        *,
        question: str,
        mode: RagMode,
        context: BuiltRagContext,
        metrics: list[GenerationAttemptMetrics],
    ) -> ValidatedRagAnswer:
        allowed_references = frozenset(entry.reference for entry in context.entries)
        response_model = BrainOnlyAnswer if mode == "brain_only" else BrainPlusModelAnswer

        def validate(value: BrainOnlyAnswer | BrainPlusModelAnswer) -> None:
            validate_rag_answer(
                value,
                mode=mode,
                allowed_references=allowed_references,
            )

        try:
            generated = await self._generator.generate_structured(
                prompt=build_rag_answer_prompt(
                    question=question,
                    knowledge_context=context.text,
                ),
                response_model=response_model,
                call_type="rag_answer",
                system_prompt=rag_system_prompt(mode),
                metrics_callback=metrics.append,
                result_validator=validate,
            )
        except StructuredOutputValidationError as error:
            raise RagInvalidAnswerError(
                "La réponse Ollama contient des citations invalides ou incohérentes."
            ) from error
        try:
            return validate_rag_answer(
                generated,
                mode=mode,
                allowed_references=allowed_references,
            )
        except StructuredOutputValidationError as error:
            raise RagInvalidAnswerError(
                "La réponse Ollama contient des citations invalides ou incohérentes."
            ) from error

    async def _revalidate_retrieval(
        self,
        retrieval: SemanticSearchResults,
    ) -> dict[UUID, KnowledgeNode]:
        if not retrieval.items:
            return {}
        expected = {item.node.id: item.node for item in retrieval.items}
        async with self._database.session_factory() as session:
            nodes = list(
                (
                    await session.scalars(
                        select(KnowledgeNode)
                        .where(KnowledgeNode.id.in_(expected))
                        .options(
                            selectinload(KnowledgeNode.source),
                            selectinload(KnowledgeNode.tag_links).selectinload(
                                KnowledgeNodeTag.tag
                            ),
                            selectinload(KnowledgeNode.evidence).selectinload(
                                KnowledgeEvidence.passage
                            ),
                        )
                    )
                )
                .unique()
                .all()
            )

        fresh = {node.id: node for node in nodes}
        if fresh.keys() != expected.keys():
            raise RagContextChangedError(
                "Une connaissance ou sa source a changé pendant la génération. "
                "Relancez la question."
            )
        for node_id, before in expected.items():
            after = fresh[node_id]
            if (
                after.source is None
                or after.source_id != before.source_id
                or semantic_text_fingerprint(title=after.title, content=after.content)
                != semantic_text_fingerprint(title=before.title, content=before.content)
            ):
                raise RagContextChangedError(
                    "Une connaissance ou sa source a changé pendant la génération. "
                    "Relancez la question."
                )
        return fresh


def _build_knowledge_results(
    retrieval: SemanticSearchResults,
    context: BuiltRagContext,
    fresh_nodes: dict[UUID, KnowledgeNode],
    *,
    used_references: frozenset[str],
) -> tuple[RagKnowledgeResult, ...]:
    references = {entry.knowledge_node_id: entry.reference for entry in context.entries}
    return tuple(
        RagKnowledgeResult(
            context_id=references.get(item.node.id),
            score=item.score,
            node=fresh_nodes[item.node.id],
            provided_to_model=item.node.id in references,
            used=references.get(item.node.id) in used_references,
        )
        for item in retrieval.items
    )


def _sum_optional(values: Iterable[int | None]) -> int | None:
    concrete = [value for value in values if value is not None]
    return sum(concrete) if concrete else None
