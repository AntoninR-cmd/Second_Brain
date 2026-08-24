from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TypeVar
from uuid import UUID

from qdrant_client import QdrantClient, models

from second_brain.vector.store import (
    StoredVectorPoint,
    VectorCollectionInfo,
    VectorPoint,
    VectorSearchHit,
    VectorStoreCompatibilityError,
    VectorStoreCorruptedError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

ResultT = TypeVar("ResultT")
logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Persistent local Qdrant isolated on one dedicated worker thread."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="second-brain-qdrant",
        )
        self._client: QdrantClient | None = None
        self._closed = False

    async def inspect_collection(self, collection_name: str) -> VectorCollectionInfo | None:
        _validate_collection_name(collection_name)
        return await self._run(
            "inspect_collection",
            lambda client: _inspect_collection(client, collection_name),
        )

    async def ensure_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> VectorCollectionInfo:
        _validate_collection_name(collection_name)
        if dimension <= 0:
            raise ValueError("la dimension vectorielle doit etre strictement positive")

        def ensure(client: QdrantClient) -> VectorCollectionInfo:
            existing = _inspect_collection(client, collection_name)
            if existing is not None:
                if existing.dimension != dimension or existing.distance != "cosine":
                    raise VectorStoreCompatibilityError(
                        f"La collection {collection_name} utilise un espace vectoriel "
                        "incompatible et doit etre reconstruite."
                    )
                return existing
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            return VectorCollectionInfo(name=collection_name, dimension=dimension)

        return await self._run("ensure_collection", ensure)

    async def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None:
        _validate_collection_name(collection_name)
        if not points:
            return
        qdrant_points = [_to_qdrant_point(point) for point in points]
        await self._run(
            "upsert",
            lambda client: client.upsert(
                collection_name=collection_name,
                points=qdrant_points,
                wait=True,
            ),
        )

    async def retrieve(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> list[StoredVectorPoint]:
        _validate_collection_name(collection_name)
        if not knowledge_node_ids:
            return []
        records = await self._run(
            "retrieve",
            lambda client: client.retrieve(
                collection_name=collection_name,
                ids=[str(node_id) for node_id in knowledge_node_ids],
                with_payload=True,
                with_vectors=False,
            ),
        )
        return [_stored_point(record.id, record.payload) for record in records]

    async def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorSearchHit]:
        _validate_collection_name(collection_name)
        vector = _validated_vector(query_vector)
        if limit <= 0:
            raise ValueError("la limite de recherche doit etre strictement positive")
        response = await self._run(
            "search",
            lambda client: client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ),
        )
        return [_search_hit(point.id, point.payload, point.score) for point in response.points]

    async def delete(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> None:
        _validate_collection_name(collection_name)
        if not knowledge_node_ids:
            return
        await self._run(
            "delete",
            lambda client: client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[str(node_id) for node_id in knowledge_node_ids]
                ),
                wait=True,
            ),
        )

    async def list_point_ids(self, collection_name: str) -> set[UUID]:
        _validate_collection_name(collection_name)

        def scroll_all(client: QdrantClient) -> set[UUID]:
            result: set[UUID] = set()
            offset: models.PointId | None = None
            while True:
                records, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
                for record in records:
                    result.add(_point_uuid(record.id))
                if next_offset is None:
                    return result
                offset = next_offset

        return await self._run("list_point_ids", scroll_all)

    async def delete_collection(self, collection_name: str) -> None:
        _validate_collection_name(collection_name)

        def delete_if_present(client: QdrantClient) -> None:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name=collection_name)

        await self._run("delete_collection", delete_if_present)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._close_client)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)

    async def _run(
        self,
        operation: str,
        callback: Callable[[QdrantClient], ResultT],
    ) -> ResultT:
        if self._closed:
            raise VectorStoreUnavailableError("L'index vectoriel local est ferme.")
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._executor,
                partial(self._invoke, callback),
            )
        except VectorStoreError:
            raise
        except Exception as error:
            logger.error(
                "Qdrant local operation failed operation=%s error_type=%s",
                operation,
                type(error).__name__,
            )
            raise _translated_error(error) from error

    def _invoke(self, callback: Callable[[QdrantClient], ResultT]) -> ResultT:
        if self._client is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
        return callback(self._client)

    def _close_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def _inspect_collection(
    client: QdrantClient,
    collection_name: str,
) -> VectorCollectionInfo | None:
    if not client.collection_exists(collection_name):
        return None
    collection = client.get_collection(collection_name=collection_name)
    vector_config = collection.config.params.vectors
    if not isinstance(vector_config, models.VectorParams):
        raise VectorStoreCompatibilityError(
            f"La collection {collection_name} utilise des vecteurs nommes non pris en charge."
        )
    if vector_config.distance != models.Distance.COSINE:
        raise VectorStoreCompatibilityError(
            f"La collection {collection_name} n'utilise pas la similarite cosinus."
        )
    return VectorCollectionInfo(
        name=collection_name,
        dimension=int(vector_config.size),
    )


def _to_qdrant_point(point: VectorPoint) -> models.PointStruct:
    fingerprint = point.fingerprint.strip()
    if not fingerprint:
        raise ValueError("le fingerprint vectoriel ne peut pas etre vide")
    return models.PointStruct(
        id=str(point.knowledge_node_id),
        vector=_validated_vector(point.vector),
        payload={
            "knowledge_node_id": str(point.knowledge_node_id),
            "source_id": str(point.source_id),
            "fingerprint": fingerprint,
        },
    )


def _stored_point(point_id: models.ExtendedPointId, payload: object) -> StoredVectorPoint:
    knowledge_node_id, source_id, fingerprint = _validated_payload(point_id, payload)
    return StoredVectorPoint(
        knowledge_node_id=knowledge_node_id,
        source_id=source_id,
        fingerprint=fingerprint,
    )


def _search_hit(
    point_id: models.ExtendedPointId,
    payload: object,
    score: float,
) -> VectorSearchHit:
    knowledge_node_id, source_id, fingerprint = _validated_payload(point_id, payload)
    if not math.isfinite(score):
        raise VectorStoreCorruptedError("Un score Qdrant n'est pas un nombre fini.")
    return VectorSearchHit(
        knowledge_node_id=knowledge_node_id,
        source_id=source_id,
        fingerprint=fingerprint,
        score=float(score),
    )


def _validated_payload(
    point_id: models.ExtendedPointId,
    payload: object,
) -> tuple[UUID, UUID, str]:
    if not isinstance(payload, dict):
        raise VectorStoreCorruptedError("Un point Qdrant ne contient aucun payload valide.")
    try:
        point_uuid = _point_uuid(point_id)
        knowledge_node_id = UUID(str(payload["knowledge_node_id"]))
        source_id = UUID(str(payload["source_id"]))
        fingerprint = str(payload["fingerprint"]).strip()
    except (KeyError, TypeError, ValueError) as error:
        raise VectorStoreCorruptedError("Un payload Qdrant est incomplet ou invalide.") from error
    if point_uuid != knowledge_node_id:
        raise VectorStoreCorruptedError(
            "L'identifiant d'un point Qdrant ne correspond pas a son payload."
        )
    if not fingerprint:
        raise VectorStoreCorruptedError("Un payload Qdrant contient un fingerprint vide.")
    return knowledge_node_id, source_id, fingerprint


def _point_uuid(point_id: models.ExtendedPointId) -> UUID:
    try:
        return UUID(str(point_id))
    except (TypeError, ValueError) as error:
        raise VectorStoreCorruptedError(
            "Qdrant contient un identifiant de point non UUID."
        ) from error


def _validated_vector(values: Sequence[float]) -> list[float]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("un vecteur doit contenir au moins une valeur")
    vector: list[float] = []
    for index, raw_value in enumerate(values):
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"vector[{index}] doit etre un nombre fini")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"vector[{index}] doit etre un nombre fini")
        vector.append(value)
    return vector


def _validate_collection_name(collection_name: str) -> None:
    if not collection_name.strip():
        raise ValueError("le nom de collection Qdrant ne peut pas etre vide")


def _translated_error(error: Exception) -> VectorStoreError:
    normalized = str(error).casefold()
    corruption_markers = (
        "corrupt",
        "deserialize",
        "invalid load key",
        "malformed",
        "unpickl",
    )
    if any(marker in normalized for marker in corruption_markers):
        return VectorStoreCorruptedError(
            "L'index Qdrant local est illisible et doit etre reconstruit."
        )
    return VectorStoreUnavailableError(
        "L'index Qdrant local est indisponible. Les donnees SQLite restent intactes."
    )
