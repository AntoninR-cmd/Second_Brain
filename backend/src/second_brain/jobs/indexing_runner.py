from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from uuid import UUID

from sqlalchemy import select

from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.session import Database
from second_brain.llm.errors import OllamaError
from second_brain.services.vector_index import (
    VECTOR_JOB_KINDS,
    VectorIndexError,
    VectorIndexService,
)
from second_brain.vector.store import VectorStoreError

logger = logging.getLogger(__name__)


class IndexingRunner:
    """Run one local, persistent vector-index job at a time."""

    def __init__(
        self,
        *,
        database: Database,
        service: VectorIndexService,
        settings: Settings,
        work_lock: asyncio.Lock | None = None,
    ) -> None:
        self._database = database
        self._service = service
        self._work_lock = work_lock
        self._wake_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._heartbeat_interval_seconds = min(
            30.0,
            max(1.0, settings.job_stale_heartbeat_seconds / 3),
        )

    async def start(self) -> None:
        if self._task is not None:
            return
        await self._recover_interrupted_jobs()
        self._task = asyncio.create_task(self._run_loop(), name="second-brain-indexing")
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

    async def enqueue(self, kind: ProcessingJobKind) -> ProcessingJob:
        job = await self._service.prepare_job(kind)
        self.wakeup()
        return job

    async def _run_loop(self) -> None:
        while True:
            self._wake_event.clear()
            job_id = await self._claim_next_job()
            if job_id is None:
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue

            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(job_id),
                name=f"second-brain-index-heartbeat-{job_id}",
            )
            outcome = "failed"
            try:
                if self._work_lock is None:
                    await self._service.run_job(job_id)
                else:
                    async with self._work_lock:
                        await self._service.run_job(job_id)
            except asyncio.CancelledError:
                raise
            except (OllamaError, VectorStoreError, VectorIndexError) as error:
                await self._mark_failed(
                    job_id,
                    code=getattr(error, "code", "vector_index_failed"),
                    error_type=type(error).__name__,
                    message=error.message,
                )
            except Exception as error:
                logger.error(
                    "Unexpected vector indexing failure processing_job_id=%s error_type=%s",
                    job_id,
                    type(error).__name__,
                )
                await self._mark_failed(
                    job_id,
                    code="internal_error",
                    error_type=type(error).__name__,
                    message="Une erreur interne a interrompu l'indexation locale.",
                )
            else:
                await self._mark_succeeded(job_id)
                outcome = "succeeded"
            finally:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task
                await self._log_benchmark(job_id, outcome=outcome)

    async def _recover_interrupted_jobs(self) -> None:
        async with self._database.session_factory() as session:
            jobs = list(
                (
                    await session.scalars(
                        select(ProcessingJob).where(
                            ProcessingJob.kind.in_(VECTOR_JOB_KINDS),
                            ProcessingJob.status == ProcessingJobStatus.RUNNING,
                        )
                    )
                ).all()
            )
            now = utc_now()
            for job in jobs:
                job.status = ProcessingJobStatus.PENDING
                job.stage = "queued"
                job.progress_message = "Indexation reprise apres le redemarrage."
                job.error_message = None
                _clear_error(job)
                job.last_activity_at = now
            await session.commit()

    async def _claim_next_job(self) -> UUID | None:
        async with self._database.session_factory() as session:
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.kind.in_(VECTOR_JOB_KINDS),
                    ProcessingJob.status == ProcessingJobStatus.PENDING,
                )
                .order_by(ProcessingJob.created_at.asc(), ProcessingJob.id.asc())
                .limit(1)
            )
            if job is None:
                return None
            now = utc_now()
            job.status = ProcessingJobStatus.RUNNING
            job.stage = "preparing"
            job.progress_message = "Preparation de l'index vectoriel."
            job.progress_current = 0
            job.progress_percent = 0
            job.attempt_count += 1
            job.started_at = job.started_at or now
            job.finished_at = None
            job.error_message = None
            _clear_error(job)
            job.last_activity_at = now
            await session.commit()
            return job.id

    async def _heartbeat_loop(self, job_id: UUID) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_seconds)
            async with self._database.session_factory() as session:
                job = await session.get(ProcessingJob, job_id)
                if job is None or job.status != ProcessingJobStatus.RUNNING:
                    return
                job.last_activity_at = utc_now()
                await session.commit()

    async def _mark_succeeded(self, job_id: UUID) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            now = utc_now()
            job.status = ProcessingJobStatus.SUCCEEDED
            job.stage = "completed"
            job.progress_current = job.progress_total
            job.progress_percent = 100
            job.progress_message = "Indexation terminee."
            job.error_message = None
            _clear_error(job)
            job.finished_at = now
            job.last_activity_at = now
            await session.commit()

    async def _mark_failed(
        self,
        job_id: UUID,
        *,
        code: str,
        error_type: str,
        message: str,
    ) -> None:
        safe_message = message.strip()[:2000] or "L'indexation locale a echoue."
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            now = utc_now()
            failure_stage = job.stage
            job.status = ProcessingJobStatus.FAILED
            job.stage = "failed"
            job.progress_message = "Indexation interrompue."
            job.error_message = safe_message
            job.error_code = code[:64]
            job.error_type = error_type[:255]
            job.error_detail = safe_message
            job.error_stage = failure_stage[:64] if failure_stage else None
            job.finished_at = now
            job.last_activity_at = now
            await session.commit()

    async def _log_benchmark(self, job_id: UUID, *, outcome: str) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
        duration_seconds = 0.0
        if job.started_at is not None and job.finished_at is not None:
            duration_seconds = max(0.0, (job.finished_at - job.started_at).total_seconds())
        average_batch_seconds = (
            job.embedding_duration_ms / 1000 / job.embedding_batch_count
            if job.embedding_batch_count
            else 0.0
        )
        logger.info(
            "Vector benchmark processing_job_id=%s kind=%s outcome=%s "
            "knowledge_nodes_completed=%d embedding_items=%d embedding_batches=%d "
            "duration_seconds=%.3f "
            "average_batch_seconds=%.3f embedding_duration_ms=%d "
            "ollama_total_duration_ns=%d prompt_eval_count=%d",
            job.id,
            job.kind.value,
            outcome,
            job.progress_current,
            job.embedding_item_count,
            job.embedding_batch_count,
            duration_seconds,
            average_batch_seconds,
            job.embedding_duration_ms,
            job.embedding_total_duration_ns,
            job.embedding_prompt_eval_count,
        )


def _clear_error(job: ProcessingJob) -> None:
    job.error_code = None
    job.error_type = None
    job.error_detail = None
    job.error_stage = None
    job.error_passage_id = None
    job.error_passage_index = None
    job.error_attempt = None
    job.error_call_type = None
