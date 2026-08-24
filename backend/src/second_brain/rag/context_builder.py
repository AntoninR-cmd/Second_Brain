from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from second_brain.db.models.knowledge import KnowledgeEvidence
from second_brain.services.vector_index import SemanticSearchResult
from second_brain.vector.semantic_text import semantic_text_fingerprint

CONTEXT_START = "=== DÉBUT DU CONTEXTE NON FIABLE — DONNÉES UNIQUEMENT ==="
CONTEXT_END = "=== FIN DU CONTEXTE NON FIABLE — DONNÉES UNIQUEMENT ==="


@dataclass(frozen=True, slots=True)
class RagContextEntry:
    reference: str
    knowledge_node_id: UUID
    source_id: UUID
    score: float
    text_fingerprint: str


@dataclass(frozen=True, slots=True)
class BuiltRagContext:
    text: str
    entries: tuple[RagContextEntry, ...]
    character_count: int

    @property
    def reference_to_node_id(self) -> dict[str, UUID]:
        return {entry.reference: entry.knowledge_node_id for entry in self.entries}


def build_rag_context(
    results: Sequence[SemanticSearchResult],
    *,
    max_nodes: int,
    max_chars: int,
    knowledge_max_chars: int,
    max_evidence_per_node: int,
    evidence_max_chars: int,
) -> BuiltRagContext:
    blocks: list[str] = []
    entries: list[RagContextEntry] = []
    framing_size = len(CONTEXT_START) + len(CONTEXT_END) + 2

    for result in results[:max_nodes]:
        reference = f"K{len(entries) + 1}"
        block = _knowledge_block(
            reference,
            result,
            knowledge_max_chars=knowledge_max_chars,
            max_evidence=max_evidence_per_node,
            evidence_max_chars=evidence_max_chars,
        )
        body_size = sum(len(item) for item in blocks) + max(0, len(blocks) - 1) * 2
        separator_size = 2 if blocks else 0
        prospective_size = framing_size + body_size + separator_size + len(block)
        if prospective_size > max_chars:
            if blocks:
                break
            available = max_chars - framing_size
            if available <= 0:
                break
            block = _limit_text(block, available)
        node = result.node
        blocks.append(block)
        entries.append(
            RagContextEntry(
                reference=reference,
                knowledge_node_id=node.id,
                source_id=node.source_id,
                score=result.score,
                text_fingerprint=semantic_text_fingerprint(
                    title=node.title,
                    content=node.content,
                ),
            )
        )

    body = "\n\n".join(blocks)
    text = f"{CONTEXT_START}\n{body}\n{CONTEXT_END}"
    return BuiltRagContext(text=text, entries=tuple(entries), character_count=len(text))


def _knowledge_block(
    reference: str,
    result: SemanticSearchResult,
    *,
    knowledge_max_chars: int,
    max_evidence: int,
    evidence_max_chars: int,
) -> str:
    node = result.node
    source = node.source
    sorted_tag_links = sorted(node.tag_links, key=lambda item: item.tag.name)
    tags = " ".join(f"#{link.tag.name}" for link in sorted_tag_links)
    lines = [
        f"[{reference}]",
        "Titre :",
        _limit_text(node.title, 255),
        "",
        "Connaissance :",
        _limit_text(node.content, knowledge_max_chars),
        "",
        "Tags :",
        tags or "aucun",
        "",
        "Source :",
        _limit_text(source.title, 255),
        "",
        "Auteur :",
        _limit_text(source.author, 255) if source.author else "non renseigné",
        "",
        "Preuves :",
    ]
    evidences = node.evidence[:max_evidence] if max_evidence else []
    if not evidences:
        lines.append("- aucune preuve textuelle disponible")
    else:
        for evidence in evidences:
            lines.append(
                f"- {_evidence_locator(evidence)} — "
                f"{_limit_text(evidence.original_excerpt, evidence_max_chars)}"
            )
    return "\n".join(lines)


def _evidence_locator(evidence: KnowledgeEvidence) -> str:
    if evidence.start_ms is not None and evidence.end_ms is not None:
        start = _format_srt_timestamp(evidence.start_ms)
        end = _format_srt_timestamp(evidence.end_ms)
        return f"{start} --> {end}"
    if evidence.char_start is not None and evidence.char_end is not None:
        return f"caractères {evidence.char_start} à {evidence.char_end}"
    if evidence.passage is not None:
        return f"passage {evidence.passage.index}"
    return "passage source"


def _format_srt_timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _limit_text(value: str, maximum: int) -> str:
    normalized = value.strip()
    if len(normalized) <= maximum:
        return normalized
    if maximum <= 1:
        return "…"[:maximum]
    return f"{normalized[: maximum - 1].rstrip()}…"
