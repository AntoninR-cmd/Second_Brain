from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from second_brain.api.dependencies import get_app_settings, get_vector_index_service
from second_brain.core.config import Settings
from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.llm.errors import (
    OllamaError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from second_brain.schemas.knowledge import (
    KnowledgeEvidenceOut,
    KnowledgeNodeSummary,
    KnowledgeSourceOut,
)
from second_brain.schemas.vector import (
    SemanticSearchItem,
    SemanticSearchProfileOut,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from second_brain.services.vector_index import (
    VectorIndexIncompatibleError,
    VectorIndexNotBuiltError,
    VectorIndexService,
)
from second_brain.vector.store import VectorStoreError

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    payload: SemanticSearchRequest,
    service: VectorIndexService = Depends(get_vector_index_service),
    settings: Settings = Depends(get_app_settings),
) -> SemanticSearchResponse:
    try:
        results = await service.search(
            payload.query,
            top_k=payload.top_k or settings.semantic_search_top_k,
        )
    except (VectorIndexNotBuiltError, VectorIndexIncompatibleError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    except OllamaModelNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    except (OllamaUnavailableError, OllamaTimeoutError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error.message,
        ) from error
    except OllamaError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error.message,
        ) from error
    except VectorStoreError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error.message,
        ) from error

    return SemanticSearchResponse(
        query=results.query,
        profile=(
            SemanticSearchProfileOut(
                model_name=results.profile.model_name,
                dimensions=results.profile.dimensions,
                distance=results.profile.distance.value,
            )
            if results.profile is not None and results.profile.dimensions is not None
            else None
        ),
        items=[
            SemanticSearchItem(
                score=item.score,
                href=f"/connaissances/{item.node.id}",
                knowledge_node=_node_summary(item.node),
                source=KnowledgeSourceOut(
                    id=item.node.source.id,
                    title=item.node.source.title,
                    type=item.node.source.type,
                    author=item.node.source.author,
                    original_filename=item.node.source.original_filename,
                    original_file_path=item.node.source.original_file_path,
                ),
                evidences=[_evidence_out(evidence) for evidence in item.node.evidence],
            )
            for item in results.items
        ],
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


def _evidence_out(evidence: KnowledgeEvidence) -> KnowledgeEvidenceOut:
    passage = evidence.passage
    return KnowledgeEvidenceOut(
        id=evidence.id,
        passage_id=evidence.passage_id,
        passage_index=passage.index,
        original_excerpt=evidence.original_excerpt,
        start_ms=evidence.start_ms,
        end_ms=evidence.end_ms,
        first_segment_index=passage.first_segment_index,
        last_segment_index=passage.last_segment_index,
        char_start=evidence.char_start,
        char_end=evidence.char_end,
    )
