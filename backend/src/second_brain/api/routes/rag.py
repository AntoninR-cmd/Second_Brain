from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from second_brain.api.dependencies import get_rag_service
from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.llm.errors import (
    OllamaError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    StructuredOutputValidationError,
)
from second_brain.rag.service import (
    RagAnswerResult,
    RagContextChangedError,
    RagInvalidAnswerError,
    RagKnowledgeResult,
    RagService,
)
from second_brain.schemas.knowledge import (
    KnowledgeEvidenceOut,
    KnowledgeNodeSummary,
    KnowledgeSourceOut,
)
from second_brain.schemas.rag import (
    RagAnswerResponse,
    RagKnowledgeOut,
    RagQuestionRequest,
    RagTimingsOut,
)
from second_brain.services.vector_index import (
    VectorIndexIncompatibleError,
    VectorIndexNotBuiltError,
)
from second_brain.vector.store import VectorStoreError

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/answer", response_model=RagAnswerResponse)
async def answer_question(
    payload: RagQuestionRequest,
    service: RagService = Depends(get_rag_service),
) -> RagAnswerResponse:
    try:
        result = await service.answer(
            payload.question,
            mode=payload.mode,
            top_k=payload.top_k,
        )
    except RagContextChangedError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    except (RagInvalidAnswerError, StructuredOutputValidationError) as error:
        message = getattr(
            error,
            "message",
            "La réponse Ollama contient des citations invalides ou incohérentes.",
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=message) from error
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

    return _answer_out(result)


def _answer_out(result: RagAnswerResult) -> RagAnswerResponse:
    return RagAnswerResponse(
        request_id=result.request_id,
        question=result.question,
        mode=result.mode,
        answer=result.answer,
        model_additions=result.model_additions,
        insufficient_context=result.insufficient_context,
        generation_model=result.generation_model,
        retrieved_knowledge=[_knowledge_out(item) for item in result.retrieved_knowledge],
        used_knowledge=[_knowledge_out(item) for item in result.used_knowledge],
        timings=RagTimingsOut(
            readiness_ms=result.timings.readiness_seconds * 1000,
            embedding_ms=result.timings.embedding_seconds * 1000,
            qdrant_ms=result.timings.qdrant_seconds * 1000,
            retrieval_sqlite_ms=result.timings.retrieval_sqlite_seconds * 1000,
            context_build_ms=result.timings.context_build_seconds * 1000,
            generation_ms=result.timings.generation_seconds * 1000,
            provenance_validation_ms=result.timings.provenance_validation_seconds * 1000,
            total_ms=result.timings.total_seconds * 1000,
            prompt_eval_count=result.timings.prompt_eval_count,
            eval_count=result.timings.eval_count,
        ),
    )


def _knowledge_out(item: RagKnowledgeResult) -> RagKnowledgeOut:
    node = item.node
    return RagKnowledgeOut(
        context_id=item.context_id,
        score=item.score,
        href=f"/connaissances/{node.id}",
        provided_to_model=item.provided_to_model,
        used=item.used,
        knowledge_node=_node_summary(node),
        source=KnowledgeSourceOut(
            id=node.source.id,
            title=node.source.title,
            type=node.source.type,
            author=node.source.author,
            original_filename=node.source.original_filename,
            original_file_path=node.source.original_file_path,
        ),
        evidences=[_evidence_out(evidence) for evidence in node.evidence],
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
