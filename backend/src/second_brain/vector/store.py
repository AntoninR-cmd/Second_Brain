from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

VectorDistance = Literal["cosine"]


class VectorStoreError(RuntimeError):
    """Base error for the reconstructible vector index."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class VectorStoreCompatibilityError(VectorStoreError):
    """The selected collection belongs to an incompatible vector space."""


class VectorStoreUnavailableError(VectorStoreError):
    """The local Qdrant store cannot currently be used."""


class VectorStoreCorruptedError(VectorStoreError):
    """The persisted index cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class VectorCollectionInfo:
    name: str
    dimension: int
    distance: VectorDistance = "cosine"


@dataclass(frozen=True, slots=True)
class VectorPoint:
    knowledge_node_id: UUID
    source_id: UUID
    fingerprint: str
    vector: Sequence[float]


@dataclass(frozen=True, slots=True)
class StoredVectorPoint:
    knowledge_node_id: UUID
    source_id: UUID
    fingerprint: str


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    knowledge_node_id: UUID
    source_id: UUID
    fingerprint: str
    score: float


class VectorStore(Protocol):
    async def inspect_collection(self, collection_name: str) -> VectorCollectionInfo | None: ...

    async def ensure_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> VectorCollectionInfo: ...

    async def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None: ...

    async def retrieve(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> list[StoredVectorPoint]: ...

    async def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorSearchHit]: ...

    async def delete(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> None: ...

    async def list_point_ids(self, collection_name: str) -> set[UUID]: ...

    async def delete_collection(self, collection_name: str) -> None: ...

    async def close(self) -> None: ...
