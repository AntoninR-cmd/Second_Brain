from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import (
    get_app_settings,
    get_indexing_runner,
    get_session,
    get_vector_index_service,
)
from second_brain.core.config import Settings
from second_brain.db.models.embedding import EmbeddingProfile
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    processing_job_is_stale,
)
from second_brain.jobs.indexing_runner import IndexingRunner
from second_brain.schemas.vector import (
    EmbeddingProfileOut,
    EmbeddingReadinessOut,
    RebuildVectorIndexRequest,
    VectorIndexStatusOut,
    VectorJobOut,
)
from second_brain.services.vector_index import (
    VECTOR_JOB_KINDS,
    VectorIndexBusyError,
    VectorIndexIncompatibleError,
    VectorIndexService,
)

router = APIRouter(prefix="/vector-index", tags=["vector-index"])


@router.get("/status", response_model=VectorIndexStatusOut)
async def vector_index_status(
    service: VectorIndexService = Depends(get_vector_index_service),
    settings: Settings = Depends(get_app_settings),
) -> VectorIndexStatusOut:
    snapshot = await service.status()
    return VectorIndexStatusOut(
        state=snapshot.state,
        configured_model=service.configured_model,
        embedding=EmbeddingReadinessOut(
            ollama_available=snapshot.readiness.ollama_available,
            configured_model=snapshot.readiness.configured_model,
            model_available=snapshot.readiness.model_available,
            error=None if snapshot.readiness.model_available else snapshot.readiness.message,
        ),
        total_nodes=snapshot.total_nodes,
        indexed_nodes=snapshot.indexed_nodes,
        pending_or_stale_nodes=snapshot.pending_or_stale_nodes,
        failed_nodes=snapshot.failed_nodes,
        orphan_points=snapshot.orphan_points,
        active_profile=_profile_out(snapshot.profile),
        active_job=(
            _job_out(snapshot.active_job, settings) if snapshot.active_job is not None else None
        ),
        error=snapshot.error,
    )


@router.post(
    "/index",
    response_model=VectorJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_knowledge(
    runner: IndexingRunner = Depends(get_indexing_runner),
    service: VectorIndexService = Depends(get_vector_index_service),
    settings: Settings = Depends(get_app_settings),
) -> VectorJobOut:
    await _require_embedding_model(service)
    try:
        job = await runner.enqueue(ProcessingJobKind.INDEX_KNOWLEDGE)
    except VectorIndexIncompatibleError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    except VectorIndexBusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    return _job_out(job, settings)


@router.post(
    "/rebuild",
    response_model=VectorJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rebuild_vector_index(
    payload: RebuildVectorIndexRequest,
    runner: IndexingRunner = Depends(get_indexing_runner),
    service: VectorIndexService = Depends(get_vector_index_service),
    settings: Settings = Depends(get_app_settings),
) -> VectorJobOut:
    del payload
    await _require_embedding_model(service)
    try:
        job = await runner.enqueue(ProcessingJobKind.REBUILD_VECTOR_INDEX)
    except VectorIndexBusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    return _job_out(job, settings)


@router.get("/jobs/{job_id}", response_model=VectorJobOut)
async def vector_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> VectorJobOut:
    job = await session.get(ProcessingJob, job_id)
    if job is None or job.kind not in VECTOR_JOB_KINDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Traitement d'indexation introuvable.",
        )
    return _job_out(job, settings)


async def _require_embedding_model(service: VectorIndexService) -> None:
    readiness = await service.get_embedding_readiness()
    if not readiness.ollama_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness.message,
        )
    if not readiness.model_available:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=readiness.message)


def _profile_out(profile: EmbeddingProfile | None) -> EmbeddingProfileOut | None:
    if profile is None:
        return None
    return EmbeddingProfileOut(
        id=profile.id,
        model_name=profile.model_name,
        dimensions=profile.dimensions,
        distance=profile.distance.value,
        logical_generation=profile.logical_generation,
    )


def _job_out(job: ProcessingJob, settings: Settings) -> VectorJobOut:
    result = VectorJobOut.model_validate(job)
    return result.model_copy(
        update={
            "is_stale": processing_job_is_stale(
                job,
                stale_after_seconds=settings.job_stale_heartbeat_seconds,
            )
        }
    )
