"""Typed interfaces and Ollama adapter for local text generation."""

from second_brain.llm.client import (
    GenerationAttemptMetrics,
    GenerationCallContext,
    GenerationCallType,
    OllamaTextGenerator,
    TextGenerator,
)
from second_brain.llm.errors import (
    OllamaError,
    OllamaHTTPError,
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    StructuredOutputValidationError,
)
from second_brain.llm.schemas import (
    KnowledgeDraft,
    OllamaReadiness,
    PassageAnalysis,
    SourceSummary,
)

__all__ = [
    "KnowledgeDraft",
    "GenerationAttemptMetrics",
    "GenerationCallContext",
    "GenerationCallType",
    "OllamaError",
    "OllamaHTTPError",
    "OllamaInvalidResponseError",
    "OllamaModelNotFoundError",
    "OllamaReadiness",
    "OllamaTextGenerator",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
    "PassageAnalysis",
    "SourceSummary",
    "StructuredOutputValidationError",
    "TextGenerator",
]
