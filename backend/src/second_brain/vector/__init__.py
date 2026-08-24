"""Typed embedding and vector-index primitives."""

from second_brain.vector.embeddings import (
    EmbeddingBatchResult,
    EmbeddingCallContext,
    EmbeddingCallMetrics,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
)
from second_brain.vector.qdrant_store import QdrantVectorStore
from second_brain.vector.semantic_text import (
    SEMANTIC_TEXT_VERSION,
    build_semantic_text,
    semantic_text_fingerprint,
)
from second_brain.vector.store import (
    StoredVectorPoint,
    VectorCollectionInfo,
    VectorPoint,
    VectorSearchHit,
    VectorStore,
    VectorStoreCompatibilityError,
    VectorStoreCorruptedError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

__all__ = [
    "EmbeddingBatchResult",
    "EmbeddingCallContext",
    "EmbeddingCallMetrics",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "QdrantVectorStore",
    "SEMANTIC_TEXT_VERSION",
    "StoredVectorPoint",
    "VectorCollectionInfo",
    "VectorPoint",
    "VectorSearchHit",
    "VectorStore",
    "VectorStoreCompatibilityError",
    "VectorStoreCorruptedError",
    "VectorStoreError",
    "VectorStoreUnavailableError",
    "build_semantic_text",
    "semantic_text_fingerprint",
]
