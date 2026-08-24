from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, field_validator

from second_brain.llm.schemas import StrictLLMSchema

RagMode = Literal["brain_only", "brain_plus_model"]
KnowledgeReference = Annotated[str, Field(pattern=r"^K[1-9][0-9]*$")]


def _normalize_required_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    if not normalized:
        raise ValueError("le texte ne peut pas être vide")
    return normalized


def _normalize_optional_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    return normalized or None


def _validate_unique_references(value: list[str]) -> list[str]:
    if len(value) != len(set(value)):
        raise ValueError("used_knowledge ne peut pas contenir de doublon")
    return value


class BrainOnlyAnswer(StrictLLMSchema):
    answer: str = Field(min_length=1, max_length=12_000)
    used_knowledge: list[KnowledgeReference] = Field(max_length=50)
    insufficient_context: bool

    _normalize_answer = field_validator("answer", mode="before")(_normalize_required_text)
    _unique_references = field_validator("used_knowledge")(_validate_unique_references)


class BrainPlusModelAnswer(StrictLLMSchema):
    from_brain: str = Field(min_length=1, max_length=12_000)
    model_additions: str | None = Field(max_length=12_000)
    used_knowledge: list[KnowledgeReference] = Field(max_length=50)
    insufficient_context: bool

    _normalize_brain_answer = field_validator("from_brain", mode="before")(_normalize_required_text)
    _normalize_model_additions = field_validator("model_additions", mode="before")(
        _normalize_optional_text
    )
    _unique_references = field_validator("used_knowledge")(_validate_unique_references)


def extract_citation_like_tokens(text: str) -> tuple[str, ...]:
    """Return bracketed K-like tokens, including malformed citations for diagnostics."""

    return tuple(match.group(1) for match in re.finditer(r"\[([Kk][^\]\r\n]*)\]", text))
