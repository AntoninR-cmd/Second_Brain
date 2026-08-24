from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from second_brain.llm.errors import StructuredOutputValidationError
from second_brain.rag.answer_schema import (
    BrainOnlyAnswer,
    BrainPlusModelAnswer,
    RagMode,
    extract_citation_like_tokens,
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "Je ne dispose pas de suffisamment d’informations dans ton second cerveau "
    "pour répondre correctement."
)
_VALID_REFERENCE = re.compile(r"^K[1-9][0-9]*$")


@dataclass(frozen=True, slots=True)
class ValidatedRagAnswer:
    brain_answer: str
    model_additions: str | None
    used_knowledge: tuple[str, ...]
    insufficient_context: bool


def validate_rag_answer(
    value: BrainOnlyAnswer | BrainPlusModelAnswer,
    *,
    mode: RagMode,
    allowed_references: frozenset[str],
) -> ValidatedRagAnswer:
    if mode == "brain_only":
        if not isinstance(value, BrainOnlyAnswer):
            raise StructuredOutputValidationError("schéma inattendu pour le mode second cerveau")
        brain_answer = value.answer
        model_additions = None
    else:
        if not isinstance(value, BrainPlusModelAnswer):
            raise StructuredOutputValidationError("schéma inattendu pour le mode mixte")
        brain_answer = value.from_brain
        model_additions = value.model_additions

    used_references = tuple(cast(list[str], value.used_knowledge))
    unknown_used = sorted(set(used_references) - allowed_references)
    if unknown_used:
        raise StructuredOutputValidationError(
            f"used_knowledge référence des connaissances inconnues : {', '.join(unknown_used)}",
            field="used_knowledge",
        )

    brain_tokens = extract_citation_like_tokens(brain_answer)
    malformed = sorted({token for token in brain_tokens if not _VALID_REFERENCE.fullmatch(token)})
    if malformed:
        raise StructuredOutputValidationError(
            f"citation mal formée dans la réponse : {', '.join(malformed)}",
            field="answer",
        )
    unknown_inline = sorted(set(brain_tokens) - allowed_references)
    if unknown_inline:
        raise StructuredOutputValidationError(
            f"citation inconnue dans la réponse : {', '.join(unknown_inline)}",
            field="answer",
        )

    if model_additions is not None:
        addition_tokens = extract_citation_like_tokens(model_additions)
        if addition_tokens:
            raise StructuredOutputValidationError(
                "model_additions ne doit contenir aucune citation K",
                field="model_additions",
            )

    if value.insufficient_context:
        if used_references or brain_tokens:
            raise StructuredOutputValidationError(
                "une réponse déclarée insuffisante ne doit citer aucune connaissance",
                field="insufficient_context",
            )
        return ValidatedRagAnswer(
            brain_answer=INSUFFICIENT_CONTEXT_ANSWER,
            model_additions=model_additions,
            used_knowledge=(),
            insufficient_context=True,
        )

    if not used_references:
        raise StructuredOutputValidationError(
            "used_knowledge doit contenir au moins une référence lorsque le contexte suffit",
            field="used_knowledge",
        )
    if set(brain_tokens) != set(used_references):
        raise StructuredOutputValidationError(
            "les citations de la réponse doivent correspondre exactement à used_knowledge",
            field="used_knowledge",
        )

    return ValidatedRagAnswer(
        brain_answer=brain_answer,
        model_additions=model_additions,
        used_knowledge=used_references,
        insufficient_context=False,
    )
