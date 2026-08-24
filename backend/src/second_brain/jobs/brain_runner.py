from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from uuid import UUID

from sqlalchemy import select

from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.models.brain import BrainProfile, BrainProfileStatus
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.session import Database
from second_brain.llm.errors import OllamaError
from second_brain.services.brain import BRAIN_JOB_KINDS, BrainError, BrainService
from second_brain.vector.store import VectorStoreError

logger = logging.getLogger(__name__)


class BrainRunner:
    """Run one persistent local brain construction or relabeling job at a time."""

    def __init__(
        self,
        *,
        database: Database,
        service: BrainService,
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
        self._task = asyncio.create_task(self._run_loop(), name="second-brain-layout")
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
                name=f"second-brain-layout-heartbeat-{job_id}",
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
            except (BrainError, OllamaError, VectorStoreError) as error:
                await self._mark_failed(
                    job_id,
                    code=getattr(error, "code", "brain_build_failed"),
                    error_type=type(error).__name__,
                    message=getattr(error, "message", str(error)),
                )
            except Exception as error:
                logger.exception(
                    "Unexpected brain construction failure processing_job_id=%s error_type=%s",
                    job_id,
                    type(error).__name__,
                )
                await self._mark_failed(
                    job_id,
                    code="internal_error",
                    error_type=type(error).__name__,
                    message="Une erreur interne a interrompu la construction du cerveau.",
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
                            ProcessingJob.kind.in_(BRAIN_JOB_KINDS),
                            ProcessingJob.status == ProcessingJobStatus.RUNNING,
                        )
                    )
                ).all()
            )
            now = utc_now()
            for job in jobs:
                job.status = ProcessingJobStatus.PENDING
                job.stage = "queued"
                job.progress_message = "Construction reprise apres le redemarrage."
                job.error_message = None
                _clear_error(job)
                job.last_activity_at = now
            await session.commit()

    async def _claim_next_job(self) -> UUID | None:
        async with self._database.session_factory() as session:
            job = await session.scalar(
                select(ProcessingJob)
                .where(
                    ProcessingJob.kind.in_(BRAIN_JOB_KINDS),
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
            job.progress_message = (
                "Preparation du cerveau mathematique."
                if job.kind == ProcessingJobKind.BUILD_BRAIN
                else "Preparation du renommage des clusters."
            )
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
            job.progress_message = (
                "Cerveau construit."
                if job.kind == ProcessingJobKind.BUILD_BRAIN
                else "Labels des clusters mis a jour."
            )
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
        safe_message = message.strip()[:2000] or "La construction du cerveau a echoue."
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            failure_stage = job.stage
            now = utc_now()
            job.status = ProcessingJobStatus.FAILED
            job.stage = "failed"
            job.progress_message = "Construction du cerveau interrompue."
            job.error_message = safe_message
            job.error_code = code[:64]
            job.error_type = error_type[:255]
            job.error_detail = safe_message
            job.error_stage = failure_stage[:64] if failure_stage else None
            job.finished_at = now
            job.last_activity_at = now
            if job.kind == ProcessingJobKind.BUILD_BRAIN and job.brain_profile_id is not None:
                profile = await session.get(BrainProfile, job.brain_profile_id)
                if profile is not None and profile.status == BrainProfileStatus.BUILDING:
                    profile.status = BrainProfileStatus.ERROR
                    profile.error_message = safe_message
            await session.commit()

    async def _log_benchmark(self, job_id: UUID, *, outcome: str) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            profile = (
                await session.get(BrainProfile, job.brain_profile_id)
                if job.brain_profile_id is not None
                else None
            )
        duration_seconds = 0.0
        if job.started_at is not None and job.finished_at is not None:
            duration_seconds = max(0.0, (job.finished_at - job.started_at).total_seconds())
        logger.info(
            "Brain job summary processing_job_id=%s profile_id=%s kind=%s outcome=%s "
            "nodes=%d edges=%d clusters=%d duration_seconds=%.3f llm_calls=%d retries=%d",
            job.id,
            job.brain_profile_id,
            job.kind.value,
            outcome,
            profile.knowledge_node_count if profile else 0,
            profile.edge_count if profile else 0,
            profile.cluster_count if profile else 0,
            duration_seconds,
            job.llm_call_count,
            job.llm_retry_count,
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
