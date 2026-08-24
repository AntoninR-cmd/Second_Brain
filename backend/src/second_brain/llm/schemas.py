from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictLLMSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class KnowledgeDraft(StrictLLMSchema):
    title: str = Field(min_length=3, max_length=160)
    content: str = Field(min_length=20, max_length=600)
    tags: list[str] = Field(default_factory=list, max_length=4)
    passage_indices: list[int] = Field(min_length=1, max_length=12)

    @field_validator("title", "content", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            raise ValueError("le texte ne peut pas etre vide")
        return normalized

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: object) -> object:
        if not isinstance(value, list):
            return value

        tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in value:
            if not isinstance(raw_tag, str):
                tags.append(raw_tag)  # type: ignore[arg-type]
                continue
            tag = unicodedata.normalize("NFKC", raw_tag).strip()
            tag = tag.removeprefix("#").strip().casefold()
            tag = re.sub(r"\s+", " ", tag)
            if not tag or tag in seen:
                continue
            if "#" in tag:
                raise ValueError("un tag ne peut pas contenir le caractere #")
            if len(tag) > 40:
                raise ValueError("un tag ne peut pas depasser 40 caracteres")
            seen.add(tag)
            tags.append(tag)
        return tags

    @field_validator("passage_indices")
    @classmethod
    def validate_passage_indices(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("les indices de passage doivent etre positifs")
        if len(value) != len(set(value)):
            raise ValueError("les indices de passage doivent etre uniques")
        return value


class PassageAnalysis(StrictLLMSchema):
    passage_index: int = Field(ge=0)
    # Ollama 0.12/qwen3.5 échoue à compiler la grammaire avec maxLength=2000,
    # tandis que 4000 est accepté. La concision est donc imposée par le prompt.
    summary: str = Field(min_length=20, max_length=4000)
    knowledge: list[KnowledgeDraft] = Field(default_factory=list, max_length=4)

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("le resume ne peut pas etre vide")
        return normalized


class SourceSummary(StrictLLMSchema):
    summary: str = Field(min_length=40, max_length=40_000)

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("le resume ne peut pas etre vide")
        return normalized


class ClusterLabel(StrictLLMSchema):
    cluster_key: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    label: str = Field(min_length=3, max_length=80)
    description: str | None = Field(default=None, min_length=10, max_length=120)

    @field_validator("label", "description", mode="before")
    @classmethod
    def normalize_cluster_text(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            raise ValueError("le texte du cluster ne peut pas etre vide")
        return normalized


class ClusterLabelBatch(StrictLLMSchema):
    labels: list[ClusterLabel] = Field(min_length=1, max_length=50)

    @field_validator("labels")
    @classmethod
    def validate_unique_cluster_keys(cls, value: list[ClusterLabel]) -> list[ClusterLabel]:
        keys = [item.cluster_key for item in value]
        if len(keys) != len(set(keys)):
            raise ValueError("les identifiants de clusters doivent etre uniques")
        return value


class OllamaReadiness(StrictLLMSchema):
    ollama_available: bool
    configured_model: str
    configured_model_digest: str | None = None
    model_available: bool
    available_models: list[str] = Field(default_factory=list)
    error_code: Literal["unavailable", "timeout", "http_error", "invalid_response"] | None = None
    message: str
