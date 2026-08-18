from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import get_session
from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.db.repositories.analysis import get_knowledge_node
from second_brain.schemas.knowledge import (
    KnowledgeEvidenceOut,
    KnowledgeNodeDetail,
    KnowledgeSourceOut,
)

router = APIRouter(prefix="/nodes", tags=["knowledge"])


@router.get("/{node_id}", response_model=KnowledgeNodeDetail)
async def knowledge_node_detail(
    node_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> KnowledgeNodeDetail:
    node = await get_knowledge_node(session, node_id)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connaissance introuvable.",
        )
    return _node_detail(node)


def _node_detail(node: KnowledgeNode) -> KnowledgeNodeDetail:
    evidences = [_evidence_out(evidence) for evidence in node.evidence]
    return KnowledgeNodeDetail(
        id=node.id,
        source_id=node.source_id,
        title=node.title,
        content=node.content,
        tags=sorted(link.tag.name for link in node.tag_links),
        evidence_count=len(evidences),
        created_at=node.created_at,
        updated_at=node.updated_at,
        source=KnowledgeSourceOut(
            id=node.source.id,
            title=node.source.title,
            type=node.source.type,
            author=node.source.author,
            original_filename=node.source.original_filename,
            original_file_path=node.source.original_file_path,
        ),
        evidences=evidences,
    )


def _evidence_out(evidence: KnowledgeEvidence) -> KnowledgeEvidenceOut:
    passage = evidence.passage
    if passage is None or evidence.passage_id is None:
        raise RuntimeError("Une preuve de connaissance ne référence aucun passage.")
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
