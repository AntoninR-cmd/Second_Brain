from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import timedelta
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.models.source import AnalysisStatus, Source
from second_brain.db.models.source_passage import SourcePassage
from second_brain.db.session import Database
from second_brain.llm.client import TextGenerator
from second_brain.llm.errors import OllamaError
from second_brain.services.analysis_pipeline import (
    AnalysisPipeline,
    AnalysisPipelineError,
    PassageProcessingError,
)

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Run one persistent local analysis job at a time."""

    def __init__(
        self,
        *,
        database: Database,
        generator: TextGenerator,
        settings: Settings,
        work_lock: asyncio.Lock | None = None,
    ) -> None:
        self._database = database
        self._work_lock = work_lock
        self._pipeline = AnalysisPipeline(
            database=database,
            generator=generator,
            settings=settings,
        )
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stale_after_seconds = settings.job_stale_heartbeat_seconds
        self._heartbeat_interval_seconds = min(
            30.0,
            max(1.0, settings.job_stale_heartbeat_seconds / 3),
        )

    async def start(self) -> None:
        if self._task is not None:
            return
        await self._recover_interrupted_jobs()
        self._task = asyncio.create_task(self._run_loop(), name="second-brain-analysis")
        self.wakeup()

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    def wakeup(self) -> None:
        self._wake_event.set()

    async def _run_loop(self) -> None:
        while True:
            self._wake_event.clear()
            claimed = await self._claim_next_job()
            if claimed is None:
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=1.0)
                # Python 3.10 distingue encore asyncio.TimeoutError du built-in
                # TimeoutError. Sans cette classe explicite, le worker mourait une
                # seconde après avoir vidé sa file.
                except asyncio.TimeoutError:
                    pass
                continue

            job_id, source_id = claimed
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(job_id),
                name=f"second-brain-heartbeat-{job_id}",
            )
            benchmark_outcome: str | None = None
            try:
                if self._work_lock is None:
                    await self._run_pipeline(job_id, source_id)
                else:
                    async with self._work_lock:
                        await self._run_pipeline(job_id, source_id)
            except asyncio.CancelledError:
                raise
            except PassageProcessingError as error:
                logger.error(
                    "Passage analysis failed exception=%s error_type=%s "
                    "source_id=%s processing_job_id=%s passage_id=%s passage_index=%s "
                    "stage=%s attempt=%s call_type=%s",
                    error.detail,
                    error.error_type,
                    source_id,
                    job_id,
                    error.passage_id,
                    error.passage_index,
                    error.stage,
                    error.attempt,
                    error.call_type,
                )
                await self._mark_failed(
                    job_id,
                    source_id,
                    error.detail,
                    error_code=error.error_code,
                    error_type=error.error_type,
                    error_detail=error.detail,
                    error_stage=error.stage,
                    error_passage_id=error.passage_id,
                    error_passage_index=error.passage_index,
                    error_attempt=error.attempt,
                    error_call_type=error.call_type,
                )
                benchmark_outcome = "failed"
            except (OllamaError, AnalysisPipelineError) as error:
                message = error.message if isinstance(error, OllamaError) else str(error)
                await self._mark_failed(
                    job_id,
                    source_id,
                    message,
                    error_code=getattr(error, "code", "analysis_pipeline_error"),
                    error_type=type(error).__name__,
                    error_detail=message,
                )
                benchmark_outcome = "failed"
            except Exception as error:
                # Ne jamais sérialiser l'exception ni sa traceback ici : une erreur de
                # persistance peut contenir les paramètres SQL et donc du texte source.
                logger.error(
                    "Unexpected failure while analyzing source %s error_type=%s",
                    source_id,
                    type(error).__name__,
                )
                await self._mark_failed(
                    job_id,
                    source_id,
                    "Une erreur interne a interrompu l'analyse locale.",
                    error_code="internal_error",
                    error_type=type(error).__name__,
                    error_detail="Une erreur interne a interrompu l'analyse locale.",
                )
                benchmark_outcome = "failed"
            else:
                await self._mark_succeeded(job_id, source_id)
                benchmark_outcome = "succeeded"
            finally:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
                if benchmark_outcome is not None:
                    await self._log_benchmark(
                        job_id,
                        source_id,
                        outcome=benchmark_outcome,
                    )

    async def _run_pipeline(self, job_id: UUID, source_id: UUID) -> None:
        await self._pipeline.run(
            source_id=source_id,
            job_id=job_id,
            progress=lambda stage, current, total, percent, message: self._update_progress(
                job_id,
                stage=stage,
                current=current,
                total=total,
                percent=percent,
                message=message,
            ),
        )

    async def _heartbeat_loop(self, job_id: UUID) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            async with self._database.session_factory() as session:
                job = await session.get(ProcessingJob, job_id)
                if job is None or job.status != ProcessingJobStatus.RUNNING:
                    return
                job.last_activity_at = utc_now()
                await session.commit()

    async def _recover_interrupted_jobs(self) -> None:
        async with self._database.session_factory() as session:
            jobs = list(
                (
                    await session.scalars(
                        select(ProcessingJob)
                        .where(
                            ProcessingJob.status == ProcessingJobStatus.RUNNING,
                            ProcessingJob.kind == ProcessingJobKind.ANALYZE_SOURCE,
                        )
                        .options(selectinload(ProcessingJob.source))
                    )
                ).all()
            )
            now = utc_now()
            stale_before = now - timedelta(seconds=self._stale_after_seconds)
            for job in jobs:
                if job.source.analysis_status == AnalysisStatus.ANALYZED:
                    job.status = ProcessingJobStatus.SUCCEEDED
                    job.stage = "completed"
                    job.progress_current = job.progress_total
                    job.progress_percent = 100
                    job.progress_message = "Analyse terminée avant le redémarrage."
                    job.finished_at = now
                    job.last_activity_at = now
                    continue
                was_stale = job.last_activity_at <= stale_before
                job.status = ProcessingJobStatus.PENDING
                job.stage = "queued"
                job.progress_message = "Analyse reprise après le redémarrage."
                job.error_message = None
                job.progress_message = (
                    "Analyse stale reprise apres le redemarrage."
                    if was_stale
                    else "Analyse reprise apres le redemarrage."
                )
                self._clear_job_diagnostic(job)
                job.source.analysis_status = AnalysisStatus.QUEUED
                job.source.analysis_error = None
                job.last_activity_at = now
            await session.commit()

    async def _claim_next_job(self) -> tuple[UUID, UUID] | None:
        async with self._database.session_factory() as session:
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.status == ProcessingJobStatus.PENDING,
                    ProcessingJob.kind == ProcessingJobKind.ANALYZE_SOURCE,
                )
                .order_by(ProcessingJob.created_at.asc(), ProcessingJob.id.asc())
                .limit(1)
            )
            if job is None:
                return None
            source = await session.get(Source, job.source_id)
            if source is None:
                job.status = ProcessingJobStatus.FAILED
                job.stage = "failed"
                job.error_message = "La source à analyser n'existe plus."
                job.finished_at = utc_now()
                await session.commit()
                return None

            now = utc_now()
            job.status = ProcessingJobStatus.RUNNING
            job.stage = "preparing"
            job.progress_message = "Préparation des passages."
            job.progress_current = 0
            job.progress_total = 0
            job.progress_percent = 0
            job.attempt_count += 1
            job.started_at = now
            job.finished_at = None
            job.error_message = None
            self._clear_job_diagnostic(job)
            job.last_activity_at = now
            source.analysis_status = AnalysisStatus.PROCESSING
            source.analysis_error = None
            source.analysis_started_at = now
            source.analysis_completed_at = None
            await session.commit()
            return job.id, source.id

    async def _update_progress(
        self,
        job_id: UUID,
        *,
        stage: str,
        current: int,
        total: int,
        percent: int,
        message: str,
    ) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.status != ProcessingJobStatus.RUNNING:
                return
            job.stage = stage
            job.progress_current = current
            job.progress_total = total
            job.progress_percent = percent
            job.progress_message = message
            job.last_activity_at = utc_now()
            await session.commit()

    async def _mark_failed(
        self,
        job_id: UUID,
        source_id: UUID,
        message: str,
        *,
        error_code: str = "analysis_failed",
        error_type: str = "AnalysisPipelineError",
        error_detail: str | None = None,
        error_stage: str | None = None,
        error_passage_id: UUID | None = None,
        error_passage_index: int | None = None,
        error_attempt: int | None = None,
        error_call_type: str | None = None,
    ) -> None:
        safe_message = message.strip()[:4000] or "L'analyse locale a échoué."
        safe_detail = (error_detail or safe_message).strip()[:4000] or safe_message
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            source = await session.get(Source, source_id)
            now = utc_now()
            if job is not None:
                failure_stage = error_stage or job.stage
                job.status = ProcessingJobStatus.FAILED
                job.stage = "failed"
                job.error_message = safe_message
                job.error_code = error_code[:64]
                job.error_type = error_type[:255]
                job.error_detail = safe_detail
                job.error_stage = failure_stage[:64] if failure_stage else None
                job.error_passage_id = error_passage_id
                job.error_passage_index = error_passage_index
                job.error_attempt = error_attempt
                job.error_call_type = error_call_type[:64] if error_call_type else None
                job.progress_message = "Analyse interrompue."
                job.finished_at = now
                job.last_activity_at = now
            if source is not None and source.analysis_status != AnalysisStatus.ANALYZED:
                source.analysis_status = AnalysisStatus.ERROR
                source.analysis_error = safe_message
                source.analysis_completed_at = now
            await session.commit()

    async def _mark_succeeded(self, job_id: UUID, source_id: UUID) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            source = await session.get(Source, source_id)
            if source is None or source.analysis_status != AnalysisStatus.ANALYZED:
                await session.rollback()
                await self._mark_failed(
                    job_id,
                    source_id,
                    "Le résultat d'analyse n'a pas été enregistré correctement.",
                )
                return
            if job is not None:
                job.status = ProcessingJobStatus.SUCCEEDED
                job.stage = "completed"
                job.progress_current = job.progress_total
                job.progress_percent = 100
                job.progress_message = "Analyse terminée."
                job.error_message = None
                self._clear_job_diagnostic(job)
                job.finished_at = utc_now()
                job.last_activity_at = job.finished_at
            await session.commit()

    async def _log_benchmark(
        self,
        job_id: UUID,
        source_id: UUID,
        *,
        outcome: str,
    ) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            passage_count = int(
                (
                    await session.scalar(
                        select(func.count(SourcePassage.id)).where(
                            SourcePassage.source_id == source_id
                        )
                    )
                )
                or 0
            )

        duration_seconds = 0.0
        if job.started_at is not None and job.finished_at is not None:
            duration_seconds = max(0.0, (job.finished_at - job.started_at).total_seconds())
        average_call_seconds = (
            job.llm_duration_ms / 1000 / job.llm_call_count if job.llm_call_count else 0.0
        )
        logger.info(
            "Analysis benchmark source_id=%s processing_job_id=%s outcome=%s "
            "source_passages=%d llm_calls=%d knowledge_nodes=%d "
            "duration_seconds=%.3f average_call_seconds=%.3f "
            "prompt_eval_count=%d eval_count=%d prompt_eval_duration_ns=%d "
            "eval_duration_ns=%d ollama_total_duration_ns=%d retries=%d",
            source_id,
            job_id,
            outcome,
            passage_count,
            job.llm_call_count,
            job.knowledge_node_count,
            duration_seconds,
            average_call_seconds,
            job.prompt_eval_count,
            job.eval_count,
            job.prompt_eval_duration_ns,
            job.eval_duration_ns,
            job.ollama_total_duration_ns,
            job.llm_retry_count,
            extra={
                "source_id": str(source_id),
                "processing_job_id": str(job_id),
                "outcome": outcome,
                "source_passages": passage_count,
                "llm_calls": job.llm_call_count,
                "knowledge_nodes": job.knowledge_node_count,
                "duration_seconds": duration_seconds,
                "average_call_seconds": average_call_seconds,
                "prompt_eval_count": job.prompt_eval_count,
                "eval_count": job.eval_count,
                "prompt_eval_duration_ns": job.prompt_eval_duration_ns,
                "eval_duration_ns": job.eval_duration_ns,
                "ollama_total_duration_ns": job.ollama_total_duration_ns,
                "retries": job.llm_retry_count,
            },
        )

    @staticmethod
    def _clear_job_diagnostic(job: ProcessingJob) -> None:
        job.error_code = None
        job.error_type = None
        job.error_detail = None
        job.error_stage = None
        job.error_passage_id = None
        job.error_passage_index = None
        job.error_attempt = None
        job.error_call_type = None
