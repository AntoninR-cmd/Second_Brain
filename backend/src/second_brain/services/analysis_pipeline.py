from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.db.models.processing import ProcessingJob
from second_brain.db.models.source import AnalysisStatus, Source, SourceType
from second_brain.db.models.source_passage import (
    SourcePassage,
    SourcePassageAnalysisStatus,
    SourcePassageSegment,
)
from second_brain.db.models.taxonomy import KnowledgeNodeTag, Tag
from second_brain.db.session import Database
from second_brain.llm.client import (
    GenerationAttemptMetrics,
    GenerationCallContext,
    GenerationCallType,
    TextGenerator,
)
from second_brain.llm.errors import (
    OllamaError,
    OllamaInvalidResponseError,
    StructuredOutputValidationError,
)
from second_brain.llm.prompt_loader import (
    build_passage_analysis_prompt,
    build_source_summary_prompt,
    system_fidelity_prompt,
)
from second_brain.llm.schemas import PassageAnalysis, SourceSummary
from second_brain.pipeline.chunking import (
    ChunkingConfig,
    SourceChunk,
    SourceSegmentInput,
    chunk_srt_segments,
    chunk_text,
    estimate_tokens,
)

ProgressCallback = Callable[[str, int, int, int, str], Awaitable[None]]
logger = logging.getLogger(__name__)


class AnalysisPipelineError(RuntimeError):
    """A safe, user-facing failure of the local analysis pipeline."""


class PassageProcessingError(AnalysisPipelineError):
    """A passage failure with safe, structured diagnostic context."""

    def __init__(
        self,
        *,
        passage_id: UUID,
        passage_index: int,
        attempt: int,
        stage: str,
        call_type: GenerationCallType,
        cause: Exception,
    ) -> None:
        self.passage_id = passage_id
        self.passage_index = passage_index
        self.attempt = attempt
        self.stage = stage
        self.call_type = call_type
        self.error_type = type(cause).__name__
        self.error_code = getattr(cause, "code", "passage_analysis_failed")
        self.detail = _safe_exception_detail(cause)
        super().__init__(self.detail)


@dataclass(frozen=True, slots=True)
class PassageContext:
    id: UUID
    chunk: SourceChunk
    first_segment_id: UUID | None
    last_segment_id: UUID | None


@dataclass(slots=True)
class KnowledgeCandidate:
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    passage_indices: set[int] = field(default_factory=set)


class AnalysisPipeline:
    def __init__(
        self,
        *,
        database: Database,
        generator: TextGenerator,
        settings: Settings,
    ) -> None:
        self._database = database
        self._generator = generator
        self._settings = settings
        self._chunking = ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_segments=settings.chunk_overlap_segments,
            pause_threshold_ms=settings.chunk_srt_pause_ms,
        )

    async def run(
        self,
        *,
        source_id: UUID,
        job_id: UUID,
        progress: ProgressCallback,
    ) -> None:
        title, passages = await self._prepare_passages(source_id)
        if not passages:
            raise AnalysisPipelineError("La source ne contient aucun passage analysable.")

        passage_total = len(passages)
        await self._sync_job_knowledge_count(job_id, source_id)
        await progress(
            "analyzing_passages",
            0,
            passage_total,
            5,
            f"Analyse des passages : 0 / {passage_total}",
        )

        passage_summaries: list[tuple[int, str]] = []
        candidates: dict[tuple[str, str], KnowledgeCandidate] = {}
        valid_indices = {passage.chunk.index for passage in passages}

        for offset, passage in enumerate(passages, start=1):
            await progress(
                "analyzing_passages",
                offset,
                passage_total,
                5 + int(70 * (offset - 1) / passage_total),
                f"Analyse des passages : {offset} / {passage_total}",
            )
            result = await self._load_completed_passage_result(
                passage,
                valid_indices=valid_indices,
                max_knowledge=self._settings.extraction_max_knowledge_per_passage,
            )
            if result is None:
                attempt = await self._mark_passage_running(passage.id)
                attempt_metrics: list[GenerationAttemptMetrics] = []

                def validate_result(value: PassageAnalysis) -> None:
                    self._validate_passage_result(
                        value,
                        expected_index=passage.chunk.index,
                        valid_indices=valid_indices,
                        max_knowledge=self._settings.extraction_max_knowledge_per_passage,
                    )

                try:
                    result = await self._generator.generate_structured(
                        prompt=build_passage_analysis_prompt(
                            source_title=title,
                            chunk=passage.chunk,
                            max_knowledge=self._settings.extraction_max_knowledge_per_passage,
                        ),
                        response_model=PassageAnalysis,
                        system_prompt=system_fidelity_prompt(),
                        call_type="passage_analysis",
                        context=GenerationCallContext(
                            source_id=source_id,
                            processing_job_id=job_id,
                            passage_id=passage.id,
                            passage_index=passage.chunk.index,
                            stage="passage_analysis",
                        ),
                        metrics_callback=attempt_metrics.append,
                        result_validator=validate_result,
                    )
                    # Fake/custom generators may not implement result_validator. Keep this
                    # domain check as a mandatory final guard before persistence.
                    validate_result(result)
                except Exception as error:
                    await self._mark_passage_failed(
                        passage.id,
                        job_id=job_id,
                        error=error,
                        metrics=attempt_metrics,
                    )
                    raise PassageProcessingError(
                        passage_id=passage.id,
                        passage_index=passage.chunk.index,
                        attempt=(attempt_metrics[-1].attempt + 1 if attempt_metrics else attempt),
                        stage="passage_analysis",
                        call_type="passage_analysis",
                        cause=error,
                    ) from error

                await self._save_completed_passage(
                    passage.id,
                    job_id=job_id,
                    result=result,
                    metrics=attempt_metrics,
                )
            passage_summaries.append((passage.chunk.index, result.summary))
            self._merge_candidates(candidates, result)
            await progress(
                "analyzing_passages",
                offset,
                passage_total,
                5 + int(70 * offset / passage_total),
                f"Analyse des passages : {offset} / {passage_total}",
            )

        await progress(
            "hierarchical_summary",
            passage_total,
            passage_total,
            80,
            "Préparation de la synthèse hiérarchique.",
        )
        final_summary = await self._build_hierarchical_summary(
            title,
            passage_summaries,
            source_id=source_id,
            job_id=job_id,
            progress=progress,
            passage_total=passage_total,
        )
        await progress(
            "saving",
            passage_total,
            passage_total,
            97,
            "Enregistrement du résumé et des connaissances.",
        )
        await self._persist_result(
            source_id=source_id,
            job_id=job_id,
            summary=final_summary,
            passages=passages,
            candidates=list(candidates.values()),
        )
        await progress(
            "completed",
            passage_total,
            passage_total,
            100,
            "Analyse terminée.",
        )

    async def _prepare_passages(
        self,
        source_id: UUID,
    ) -> tuple[str, list[PassageContext]]:
        async with self._database.session_factory() as session:
            source = await session.scalar(
                select(Source).where(Source.id == source_id).options(selectinload(Source.segments))
            )
            if source is None:
                raise AnalysisPipelineError("La source à analyser n'existe plus.")

            existing = await self._load_passage_contexts(session, source_id)
            if existing:
                return source.title, existing

            if source.type == SourceType.SRT:
                chunks = chunk_srt_segments(
                    [
                        SourceSegmentInput(
                            index=segment.index,
                            text=segment.text,
                            start_ms=segment.start_ms,
                            end_ms=segment.end_ms,
                        )
                        for segment in source.segments
                    ],
                    self._chunking,
                )
            else:
                chunks = chunk_text(source.raw_text, self._chunking)

            segments_by_index = {segment.index: segment for segment in source.segments}
            for chunk in chunks:
                passage = SourcePassage(
                    source_id=source.id,
                    index=chunk.index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    first_segment_index=(
                        chunk.segment_indices[0] if chunk.segment_indices else None
                    ),
                    last_segment_index=(
                        chunk.segment_indices[-1] if chunk.segment_indices else None
                    ),
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )
                session.add(passage)
                for position, segment_index in enumerate(chunk.segment_indices):
                    segment = segments_by_index.get(segment_index)
                    if segment is None:
                        raise AnalysisPipelineError(
                            "Un passage SRT référence un segment source introuvable."
                        )
                    passage.segment_links.append(
                        SourcePassageSegment(segment_id=segment.id, position=position)
                    )
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise

            return source.title, await self._load_passage_contexts(session, source_id)

    async def _load_passage_contexts(
        self,
        session: AsyncSession,
        source_id: UUID,
    ) -> list[PassageContext]:
        passages = list(
            (
                await session.scalars(
                    select(SourcePassage)
                    .where(SourcePassage.source_id == source_id)
                    .options(
                        selectinload(SourcePassage.segment_links).selectinload(
                            SourcePassageSegment.segment
                        )
                    )
                    .order_by(SourcePassage.index.asc())
                )
            )
            .unique()
            .all()
        )
        contexts: list[PassageContext] = []
        for passage in passages:
            segments = [link.segment for link in passage.segment_links]
            starts = [segment.start_ms for segment in segments if segment.start_ms is not None]
            ends = [segment.end_ms for segment in segments if segment.end_ms is not None]
            contexts.append(
                PassageContext(
                    id=passage.id,
                    chunk=SourceChunk(
                        index=passage.index,
                        text=passage.text,
                        token_count=passage.token_count,
                        segment_indices=tuple(segment.index for segment in segments),
                        char_start=passage.char_start,
                        char_end=passage.char_end,
                        start_ms=min(starts) if starts else None,
                        end_ms=max(ends) if ends else None,
                    ),
                    first_segment_id=segments[0].id if segments else None,
                    last_segment_id=segments[-1].id if segments else None,
                )
            )
        return contexts

    async def _load_completed_passage_result(
        self,
        passage_context: PassageContext,
        *,
        valid_indices: set[int],
        max_knowledge: int,
    ) -> PassageAnalysis | None:
        async with self._database.session_factory() as session:
            passage = await session.get(SourcePassage, passage_context.id)
            if passage is None:
                raise AnalysisPipelineError("Un passage d'analyse n'existe plus.")
            if passage.analysis_status != SourcePassageAnalysisStatus.COMPLETED:
                return None

            try:
                if passage.analysis_payload_json is None:
                    raise ValueError("payload absent")
                result = PassageAnalysis.model_validate_json(passage.analysis_payload_json)
                self._validate_passage_result(
                    result,
                    expected_index=passage_context.chunk.index,
                    valid_indices=valid_indices,
                    max_knowledge=max_knowledge,
                )
            except (
                ValidationError,
                ValueError,
                StructuredOutputValidationError,
                AnalysisPipelineError,
            ):
                # A schema change or manual DB alteration must never let an invalid
                # checkpoint contaminate the final nodes. Recompute only this passage.
                passage.analysis_status = SourcePassageAnalysisStatus.FAILED
                passage.analysis_payload_json = None
                passage.intermediate_summary = None
                passage.analysis_error = (
                    "Le resultat intermediaire persiste est invalide et sera recalcule."
                )
                passage.analysis_completed_at = None
                passage.analysis_last_activity_at = utc_now()
                await session.commit()
                return None
            return result

    async def _mark_passage_running(self, passage_id: UUID) -> int:
        async with self._database.session_factory() as session:
            passage = await session.get(SourcePassage, passage_id)
            if passage is None:
                raise AnalysisPipelineError("Un passage d'analyse n'existe plus.")
            if passage.analysis_status == SourcePassageAnalysisStatus.COMPLETED:
                raise AnalysisPipelineError(
                    "Un passage valide ne peut pas etre relance sans invalidation explicite."
                )
            now = utc_now()
            passage.analysis_status = SourcePassageAnalysisStatus.RUNNING
            passage.analysis_attempt_count += 1
            passage.analysis_started_at = now
            passage.analysis_completed_at = None
            passage.analysis_last_activity_at = now
            passage.analysis_payload_json = None
            passage.intermediate_summary = None
            passage.analysis_error = None
            passage.knowledge_count = 0
            await session.commit()
            return passage.analysis_attempt_count

    async def _mark_passage_failed(
        self,
        passage_id: UUID,
        *,
        job_id: UUID,
        error: Exception,
        metrics: Sequence[GenerationAttemptMetrics],
    ) -> None:
        async with self._database.session_factory() as session:
            passage = await session.get(SourcePassage, passage_id)
            if passage is None:
                raise AnalysisPipelineError("Un passage d'analyse n'existe plus.")
            passage.analysis_status = SourcePassageAnalysisStatus.FAILED
            passage.analysis_error = _safe_exception_detail(error)[:4000]
            passage.analysis_payload_json = None
            passage.intermediate_summary = None
            passage.analysis_completed_at = None
            passage.analysis_last_activity_at = utc_now()
            self._add_metrics(passage, metrics)
            await self._add_job_metrics(session, job_id, metrics)
            await session.commit()

    async def _save_completed_passage(
        self,
        passage_id: UUID,
        *,
        job_id: UUID,
        result: PassageAnalysis,
        metrics: Sequence[GenerationAttemptMetrics],
    ) -> None:
        async with self._database.session_factory() as session:
            passage = await session.get(SourcePassage, passage_id)
            if passage is None:
                raise AnalysisPipelineError("Un passage d'analyse n'existe plus.")
            now = utc_now()
            passage.analysis_status = SourcePassageAnalysisStatus.COMPLETED
            passage.analysis_payload_json = result.model_dump_json()
            passage.intermediate_summary = result.summary
            passage.analysis_error = None
            passage.analysis_completed_at = now
            passage.analysis_last_activity_at = now
            passage.knowledge_count = len(result.knowledge)
            self._add_metrics(passage, metrics)
            await self._add_job_metrics(session, job_id, metrics)
            await session.flush()
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise AnalysisPipelineError("Le traitement d'analyse n'existe plus.")
            validated_knowledge_count = await session.scalar(
                select(func.sum(SourcePassage.knowledge_count)).where(
                    SourcePassage.source_id == passage.source_id,
                    SourcePassage.analysis_status == SourcePassageAnalysisStatus.COMPLETED,
                )
            )
            job.knowledge_node_count = int(validated_knowledge_count or 0)
            await session.commit()
            logger.info(
                "Passage checkpoint source_id=%s processing_job_id=%s passage_id=%s "
                "passage_index=%s knowledge_count=%d llm_calls=%d retries=%d "
                "prompt_eval_count=%d eval_count=%d",
                passage.source_id,
                job_id,
                passage.id,
                passage.index,
                passage.knowledge_count,
                len(metrics),
                sum(metric.attempt > 0 for metric in metrics),
                sum(metric.prompt_eval_count or 0 for metric in metrics),
                sum(metric.eval_count or 0 for metric in metrics),
                extra={
                    "source_id": str(passage.source_id),
                    "processing_job_id": str(job_id),
                    "passage_id": str(passage.id),
                    "passage_index": passage.index,
                    "knowledge_count": passage.knowledge_count,
                    "llm_calls": len(metrics),
                    "retries": sum(metric.attempt > 0 for metric in metrics),
                    "prompt_eval_count": sum(metric.prompt_eval_count or 0 for metric in metrics),
                    "eval_count": sum(metric.eval_count or 0 for metric in metrics),
                },
            )

    @staticmethod
    def _add_metrics(
        target: SourcePassage | ProcessingJob,
        metrics: Sequence[GenerationAttemptMetrics],
    ) -> None:
        if not metrics:
            return
        target.llm_call_count += len(metrics)
        target.llm_retry_count += sum(metric.attempt > 0 for metric in metrics)
        target.llm_duration_ms += round(sum(metric.duration_seconds for metric in metrics) * 1000)
        target.ollama_total_duration_ns += sum(metric.total_duration_ns or 0 for metric in metrics)
        target.prompt_eval_count += sum(metric.prompt_eval_count or 0 for metric in metrics)
        target.prompt_eval_duration_ns += sum(
            metric.prompt_eval_duration_ns or 0 for metric in metrics
        )
        target.eval_count += sum(metric.eval_count or 0 for metric in metrics)
        target.eval_duration_ns += sum(metric.eval_duration_ns or 0 for metric in metrics)

    async def _add_job_metrics(
        self,
        session: AsyncSession,
        job_id: UUID,
        metrics: Sequence[GenerationAttemptMetrics],
    ) -> None:
        job = await session.get(ProcessingJob, job_id)
        if job is None:
            raise AnalysisPipelineError("Le traitement d'analyse n'existe plus.")
        self._add_metrics(job, metrics)

    async def _save_job_metrics(
        self,
        job_id: UUID,
        metrics: Sequence[GenerationAttemptMetrics],
    ) -> None:
        if not metrics:
            return
        async with self._database.session_factory() as session:
            await self._add_job_metrics(session, job_id, metrics)
            await session.commit()

    async def _sync_job_knowledge_count(self, job_id: UUID, source_id: UUID) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise AnalysisPipelineError("Le traitement d'analyse n'existe plus.")
            validated_knowledge_count = await session.scalar(
                select(func.sum(SourcePassage.knowledge_count)).where(
                    SourcePassage.source_id == source_id,
                    SourcePassage.analysis_status == SourcePassageAnalysisStatus.COMPLETED,
                )
            )
            job.knowledge_node_count = int(validated_knowledge_count or 0)
            await session.commit()

    @staticmethod
    def _validate_passage_result(
        result: PassageAnalysis,
        *,
        expected_index: int,
        valid_indices: set[int],
        max_knowledge: int,
    ) -> None:
        if result.passage_index != expected_index:
            raise StructuredOutputValidationError(
                f"passage_index: attendu {expected_index}, recu {result.passage_index}.",
                field="passage_index",
            )
        if len(result.knowledge) > max_knowledge:
            raise StructuredOutputValidationError(
                "knowledge: "
                f"au plus {max_knowledge} element(s) attendu(s), "
                f"recu {len(result.knowledge)}.",
                field="knowledge",
            )
        for knowledge_index, knowledge in enumerate(result.knowledge):
            references = set(knowledge.passage_indices)
            if references != {expected_index} or not references <= valid_indices:
                raise StructuredOutputValidationError(
                    "provenance invalide : "
                    f"knowledge[{knowledge_index}].passage_indices: "
                    f"attendu [{expected_index}], recu {knowledge.passage_indices}.",
                    field=f"knowledge[{knowledge_index}].passage_indices",
                )

    @staticmethod
    def _merge_candidates(
        candidates: dict[tuple[str, str], KnowledgeCandidate],
        result: PassageAnalysis,
    ) -> None:
        for draft in result.knowledge:
            key = (_identity(draft.title), _identity(draft.content))
            candidate = candidates.get(key)
            if candidate is None:
                candidate = KnowledgeCandidate(title=draft.title, content=draft.content)
                candidates[key] = candidate
            candidate.passage_indices.update(draft.passage_indices)
            for tag in draft.tags:
                if tag not in candidate.tags and len(candidate.tags) < 8:
                    candidate.tags.append(tag)

    async def _build_hierarchical_summary(
        self,
        title: str,
        summaries: Sequence[tuple[int, str]],
        *,
        source_id: UUID,
        job_id: UUID,
        progress: ProgressCallback,
        passage_total: int,
    ) -> str:
        budget = max(256, min(self._settings.ollama_num_ctx // 2, 3000))
        current = self._split_oversized_summaries(summaries, budget)
        summary_percent = 80
        for level in range(8):
            groups = _group_summaries(current, budget)
            generated: list[tuple[int, str]] = []
            for index, group in enumerate(groups):
                is_final_call = len(groups) == 1
                stage = "final_summary" if is_final_call else "hierarchical_summary"
                if is_final_call:
                    summary_percent = max(summary_percent, 90)
                    message = "Génération du résumé final."
                    call_type = "final_summary"
                else:
                    message = (
                        f"Synthèse hiérarchique : niveau {level + 1}, "
                        f"groupe {index + 1} / {len(groups)}."
                    )
                    call_type = "hierarchical_summary"
                await progress(
                    stage,
                    passage_total,
                    passage_total,
                    summary_percent,
                    message,
                )
                attempt_metrics: list[GenerationAttemptMetrics] = []
                try:
                    result = await self._generator.generate_structured(
                        prompt=build_source_summary_prompt(
                            source_title=title,
                            passage_summaries=group,
                        ),
                        response_model=SourceSummary,
                        system_prompt=system_fidelity_prompt(),
                        call_type=call_type,
                        context=GenerationCallContext(
                            source_id=source_id,
                            processing_job_id=job_id,
                            stage=stage,
                        ),
                        metrics_callback=attempt_metrics.append,
                    )
                finally:
                    await self._save_job_metrics(job_id, attempt_metrics)
                generated.append((index, result.summary))
                summary_percent = min(
                    95,
                    summary_percent + (5 if is_final_call else 1),
                )
                await progress(
                    stage,
                    passage_total,
                    passage_total,
                    summary_percent,
                    message,
                )
            if len(generated) == 1:
                return generated[0][1]
            current = self._split_oversized_summaries(generated, budget)
        raise AnalysisPipelineError(
            "La synthèse hiérarchique n'a pas convergé dans la limite prévue."
        )

    @staticmethod
    def _split_oversized_summaries(
        summaries: Sequence[tuple[int, str]],
        budget: int,
    ) -> list[tuple[int, str]]:
        result: list[tuple[int, str]] = []
        config = ChunkingConfig(
            target_tokens=max(1, int(budget * 0.75)),
            max_tokens=budget,
            overlap_segments=0,
        )
        for _index, summary in summaries:
            if estimate_tokens(summary) <= budget:
                result.append((len(result), summary))
                continue
            for chunk in chunk_text(summary, config):
                result.append((len(result), chunk.text))
        return result

    async def _persist_result(
        self,
        *,
        source_id: UUID,
        job_id: UUID,
        summary: str,
        passages: Sequence[PassageContext],
        candidates: Sequence[KnowledgeCandidate],
    ) -> None:
        async with self._database.session_factory() as session:
            source = await session.get(Source, source_id)
            if source is None:
                raise AnalysisPipelineError("La source analysée n'existe plus.")
            existing_count = await session.scalar(
                select(func.count(KnowledgeNode.id)).where(KnowledgeNode.source_id == source_id)
            )
            if existing_count:
                raise AnalysisPipelineError(
                    "Des connaissances existent déjà pour cette source ; "
                    "aucun doublon n'a été créé."
                )

            passage_by_index = {passage.chunk.index: passage for passage in passages}
            tag_names = sorted({tag for candidate in candidates for tag in candidate.tags})
            tags_by_name: dict[str, Tag] = {}
            if tag_names:
                existing_tags = list(
                    (
                        await session.scalars(select(Tag).where(Tag.normalized_name.in_(tag_names)))
                    ).all()
                )
                tags_by_name = {tag.normalized_name: tag for tag in existing_tags}
                for name in tag_names:
                    if name not in tags_by_name:
                        tag = Tag(name=name, normalized_name=name)
                        session.add(tag)
                        tags_by_name[name] = tag

            for candidate in candidates:
                node = KnowledgeNode(
                    source_id=source_id,
                    title=candidate.title,
                    content=candidate.content,
                )
                session.add(node)
                for name in candidate.tags:
                    node.tag_links.append(KnowledgeNodeTag(tag=tags_by_name[name]))
                for evidence_index, passage_index in enumerate(sorted(candidate.passage_indices)):
                    passage = passage_by_index.get(passage_index)
                    if passage is None:
                        raise AnalysisPipelineError("Une connaissance référence un passage absent.")
                    node.evidence.append(
                        KnowledgeEvidence(
                            source_id=source_id,
                            passage_id=passage.id,
                            evidence_index=evidence_index,
                            first_segment_id=passage.first_segment_id,
                            last_segment_id=passage.last_segment_id,
                            original_excerpt=passage.chunk.text,
                            start_ms=passage.chunk.start_ms,
                            end_ms=passage.chunk.end_ms,
                            char_start=passage.chunk.char_start,
                            char_end=passage.chunk.char_end,
                        )
                    )

            source.summary = summary
            source.analysis_status = AnalysisStatus.ANALYZED
            source.analysis_error = None
            source.analysis_completed_at = utc_now()
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise AnalysisPipelineError("Le traitement d'analyse n'existe plus.")
            job.knowledge_node_count = len(candidates)
            try:
                await session.commit()
            except Exception:
                await session.rollback()
                raise


def _identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _safe_exception_detail(error: Exception) -> str:
    if isinstance(error, OllamaInvalidResponseError):
        detail = error.message.strip()
        if error.detail:
            detail = f"{detail} Detail : {error.detail.strip()}"
        return detail[:4000]
    if isinstance(error, OllamaError):
        return error.message.strip()[:4000]
    if isinstance(error, StructuredOutputValidationError):
        return error.detail.strip()[:4000]
    if isinstance(error, AnalysisPipelineError):
        return str(error).strip()[:4000]
    return "Une erreur interne a interrompu l'analyse locale."


def _group_summaries(
    summaries: Sequence[tuple[int, str]],
    budget: int,
) -> list[list[tuple[int, str]]]:
    groups: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    current_tokens = 0
    for item in summaries:
        item_tokens = estimate_tokens(item[1]) + 8
        if current and current_tokens + item_tokens > budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens
    if current:
        groups.append(current)
    return groups
