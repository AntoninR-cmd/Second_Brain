from __future__ import annotations

from typing import cast

import pytest
from second_brain.core.config import Settings
from second_brain.db.migrations import migrate_database
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.models.source import AnalysisStatus, Source
from second_brain.db.session import Database
from second_brain.jobs.analysis_runner import AnalysisRunner
from second_brain.llm.client import TextGenerator


def _runner(database: Database, settings: Settings) -> AnalysisRunner:
    return AnalysisRunner(
        database=database,
        generator=cast(TextGenerator, object()),
        settings=settings,
    )


@pytest.mark.anyio
async def test_analysis_runner_claims_only_analysis_jobs(settings: Settings) -> None:
    await migrate_database(settings.resolved_database_url)
    database = Database(settings.resolved_database_url)
    try:
        async with database.session_factory() as session:
            source = Source(title="Source a analyser", raw_text="Texte source")
            session.add(source)
            await session.flush()
            vector_job = ProcessingJob(
                source_id=None,
                kind=ProcessingJobKind.INDEX_KNOWLEDGE,
                status=ProcessingJobStatus.PENDING,
                stage="queued",
            )
            analysis_job = ProcessingJob(
                source_id=source.id,
                kind=ProcessingJobKind.ANALYZE_SOURCE,
                status=ProcessingJobStatus.PENDING,
                stage="queued",
            )
            session.add_all([vector_job, analysis_job])
            await session.commit()
            vector_job_id = vector_job.id
            analysis_job_id = analysis_job.id
            source_id = source.id

        claimed = await _runner(database, settings)._claim_next_job()

        assert claimed == (analysis_job_id, source_id)
        async with database.session_factory() as session:
            untouched_vector_job = await session.get(ProcessingJob, vector_job_id)
            assert untouched_vector_job is not None
            assert untouched_vector_job.status == ProcessingJobStatus.PENDING
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_analysis_runner_recovery_ignores_vector_jobs(settings: Settings) -> None:
    await migrate_database(settings.resolved_database_url)
    database = Database(settings.resolved_database_url)
    try:
        async with database.session_factory() as session:
            source = Source(
                title="Source interrompue",
                raw_text="Texte source",
                analysis_status=AnalysisStatus.PROCESSING,
            )
            session.add(source)
            await session.flush()
            analysis_job = ProcessingJob(
                source_id=source.id,
                kind=ProcessingJobKind.ANALYZE_SOURCE,
                status=ProcessingJobStatus.RUNNING,
                stage="analyzing_passages",
            )
            vector_job = ProcessingJob(
                source_id=None,
                kind=ProcessingJobKind.REBUILD_VECTOR_INDEX,
                status=ProcessingJobStatus.RUNNING,
                stage="embedding",
            )
            session.add_all([analysis_job, vector_job])
            await session.commit()
            analysis_job_id = analysis_job.id
            vector_job_id = vector_job.id

        await _runner(database, settings)._recover_interrupted_jobs()

        async with database.session_factory() as session:
            recovered_analysis = await session.get(ProcessingJob, analysis_job_id)
            untouched_vector = await session.get(ProcessingJob, vector_job_id)
            assert recovered_analysis is not None
            assert recovered_analysis.status == ProcessingJobStatus.PENDING
            assert untouched_vector is not None
            assert untouched_vector.status == ProcessingJobStatus.RUNNING
    finally:
        await database.dispose()
