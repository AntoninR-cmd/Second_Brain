from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import (
    get_analysis_runner,
    get_app_settings,
    get_session,
    get_text_generator,
)
from second_brain.core.config import Settings
from second_brain.db.models.knowledge import KnowledgeNode
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    processing_job_is_stale,
)
from second_brain.db.repositories.analysis import (
    SourceAlreadyAnalyzedError,
    enqueue_source_analysis,
    get_latest_source_analysis_job,
    get_processing_job,
    list_source_knowledge,
)
from second_brain.db.repositories.sources import get_source
from second_brain.jobs.analysis_runner import AnalysisRunner
from second_brain.llm.client import TextGenerator
from second_brain.schemas.analysis import AnalysisJobOut
from second_brain.schemas.knowledge import KnowledgeNodeList, KnowledgeNodeSummary

router = APIRouter(tags=["analysis"])


@router.post(
    "/sources/{source_id}/analyze",
    response_model=AnalysisJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def analyze_source(
    source_id: UUID,
    session: AsyncSession = Depends(get_session),
    generator: TextGenerator = Depends(get_text_generator),
    runner: AnalysisRunner = Depends(get_analysis_runner),
    settings: Settings = Depends(get_app_settings),
) -> AnalysisJobOut:
    source = await get_source(session, source_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source introuvable.")

    readiness = await generator.get_readiness()
    if not readiness.ollama_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness.message,
        )
    if not readiness.model_available:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=readiness.message)

    try:
        job = await enqueue_source_analysis(session, source)
    except SourceAlreadyAnalyzedError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    runner.wakeup()
    return _job_out(job, settings)


@router.get("/jobs/{job_id}", response_model=AnalysisJobOut)
async def analysis_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> AnalysisJobOut:
    job = await get_processing_job(session, job_id)
    if job is None or job.kind != ProcessingJobKind.ANALYZE_SOURCE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Traitement introuvable.",
        )
    return _job_out(job, settings)


@router.get("/sources/{source_id}/analysis", response_model=AnalysisJobOut)
async def latest_source_analysis(
    source_id: UUID,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> AnalysisJobOut:
    source = await get_source(session, source_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source introuvable.")
    job = await get_latest_source_analysis_job(session, source_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun traitement d'analyse pour cette source.",
        )
    return _job_out(job, settings)


@router.get("/sources/{source_id}/nodes", response_model=KnowledgeNodeList)
async def source_knowledge_nodes(
    source_id: UUID,
    limit: int = Query(default=100, ge=1, le=200),
    cursor: UUID | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> KnowledgeNodeList:
    source = await get_source(session, source_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source introuvable.")
    try:
        nodes, next_cursor = await list_source_knowledge(
            session,
            source_id=source_id,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le curseur de pagination est inconnu.",
        ) from error
    return KnowledgeNodeList(
        items=[_node_summary(node) for node in nodes],
        next_cursor=next_cursor,
    )


def _node_summary(node: KnowledgeNode) -> KnowledgeNodeSummary:
    return KnowledgeNodeSummary(
        id=node.id,
        source_id=node.source_id,
        title=node.title,
        content=node.content,
        tags=sorted(link.tag.name for link in node.tag_links),
        evidence_count=len(node.evidence),
        created_at=node.created_at,
        updated_at=node.updated_at,
    )


def _job_out(job: ProcessingJob, settings: Settings) -> AnalysisJobOut:
    result = AnalysisJobOut.model_validate(job)
    return result.model_copy(
        update={
            "is_stale": processing_job_is_stale(
                job,
                stale_after_seconds=settings.job_stale_heartbeat_seconds,
            )
        }
    )
