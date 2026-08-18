from __future__ import annotations

from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.models.source import AnalysisStatus, Source
from second_brain.db.models.taxonomy import KnowledgeNodeTag


class SourceAlreadyAnalyzedError(ValueError):
    pass


async def enqueue_source_analysis(
    session: AsyncSession,
    source: Source,
) -> ProcessingJob:
    active_job = await session.scalar(
        select(ProcessingJob)
        .where(
            ProcessingJob.source_id == source.id,
            ProcessingJob.kind == ProcessingJobKind.ANALYZE_SOURCE,
            ProcessingJob.status.in_([ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING]),
        )
        .order_by(ProcessingJob.created_at.desc())
        .limit(1)
    )
    if active_job is not None:
        return active_job
    if source.analysis_status == AnalysisStatus.ANALYZED:
        raise SourceAlreadyAnalyzedError("Cette source a déjà été analysée.")

    job = ProcessingJob(
        source_id=source.id,
        kind=ProcessingJobKind.ANALYZE_SOURCE,
        status=ProcessingJobStatus.PENDING,
        stage="queued",
        progress_current=0,
        progress_total=1,
        progress_message="Analyse en attente.",
    )
    source.analysis_status = AnalysisStatus.QUEUED
    source.analysis_error = None
    source.analysis_started_at = None
    source.analysis_completed_at = None
    session.add(job)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    await session.refresh(job)
    return job


async def get_processing_job(
    session: AsyncSession,
    job_id: UUID,
) -> ProcessingJob | None:
    return await session.get(ProcessingJob, job_id)


async def get_latest_source_analysis_job(
    session: AsyncSession,
    source_id: UUID,
) -> ProcessingJob | None:
    return await session.scalar(
        select(ProcessingJob)
        .where(
            ProcessingJob.source_id == source_id,
            ProcessingJob.kind == ProcessingJobKind.ANALYZE_SOURCE,
        )
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
        .limit(1)
    )


async def count_source_knowledge(session: AsyncSession, source_id: UUID) -> int:
    count = await session.scalar(
        select(func.count(KnowledgeNode.id)).where(KnowledgeNode.source_id == source_id)
    )
    return int(count or 0)


async def list_source_knowledge(
    session: AsyncSession,
    *,
    source_id: UUID,
    limit: int,
    cursor: UUID | None,
) -> tuple[list[KnowledgeNode], UUID | None]:
    statement = select(KnowledgeNode).where(KnowledgeNode.source_id == source_id)
    if cursor is not None:
        boundary = await session.get(KnowledgeNode, cursor)
        if boundary is None or boundary.source_id != source_id:
            raise ValueError("Unknown knowledge cursor")
        statement = statement.where(
            or_(
                KnowledgeNode.created_at < boundary.created_at,
                and_(
                    KnowledgeNode.created_at == boundary.created_at,
                    KnowledgeNode.id < boundary.id,
                ),
            )
        )
    statement = statement.options(
        selectinload(KnowledgeNode.tag_links).selectinload(KnowledgeNodeTag.tag),
        selectinload(KnowledgeNode.evidence),
    ).order_by(KnowledgeNode.created_at.desc(), KnowledgeNode.id.desc())

    rows = list((await session.scalars(statement.limit(limit + 1))).unique().all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    return items, next_cursor


async def get_knowledge_node(
    session: AsyncSession,
    node_id: UUID,
) -> KnowledgeNode | None:
    return await session.scalar(
        select(KnowledgeNode)
        .where(KnowledgeNode.id == node_id)
        .options(
            selectinload(KnowledgeNode.source),
            selectinload(KnowledgeNode.tag_links).selectinload(KnowledgeNodeTag.tag),
            selectinload(KnowledgeNode.evidence).selectinload(KnowledgeEvidence.passage),
        )
    )
