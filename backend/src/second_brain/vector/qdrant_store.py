from __future__ import annotations

import asyncio
import json
import logging
import math
import pickle
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TypeVar
from uuid import UUID, uuid4

from qdrant_client import QdrantClient, models

from second_brain.vector.store import (
    StoredVector,
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
_OWNER_MARKER = ".second-brain-qdrant"


class QdrantVectorStore:
    """Persistent local Qdrant isolated on one dedicated worker thread."""

    def __init__(self, path: Path, *, reset_root: Path | None = None) -> None:
        self._path = path.resolve()
        self._reset_root = (reset_root or self._path.parent).resolve()
        try:
            self._path.relative_to(self._reset_root)
        except ValueError as error:
            raise ValueError("Le stockage Qdrant doit rester dans sa racine dediee.") from error
        if self._path == self._reset_root:
            raise ValueError("Le stockage Qdrant doit etre un sous-dossier dedie.")
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

    async def retrieve_vectors(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> list[StoredVector]:
        """Read existing vectors in bounded batches without exposing Qdrant upstream."""

        _validate_collection_name(collection_name)
        if not knowledge_node_ids:
            return []
        ordered_ids = list(dict.fromkeys(knowledge_node_ids))

        def retrieve_all(client: QdrantClient) -> list[StoredVector]:
            result: list[StoredVector] = []
            for start in range(0, len(ordered_ids), 256):
                records = client.retrieve(
                    collection_name=collection_name,
                    ids=[str(node_id) for node_id in ordered_ids[start : start + 256]],
                    with_payload=True,
                    with_vectors=True,
                )
                result.extend(
                    _stored_vector(record.id, record.payload, record.vector) for record in records
                )
            return result

        return await self._run("retrieve_vectors", retrieve_all)

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

    async def reset_storage(self) -> None:
        """Atomically replace only the reconstructible local Qdrant directory."""

        if self._closed:
            raise VectorStoreUnavailableError("L'index vectoriel local est ferme.")
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._reset_storage)
        except VectorStoreError:
            raise
        except Exception as error:
            logger.error(
                "Qdrant local reset failed error_type=%s",
                type(error).__name__,
            )
            raise _translated_error(error) from error

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
            self._assert_adoptable_path()
            candidate = QdrantClient(path=str(self._path))
            try:
                candidate.get_collections()
                (self._path / _OWNER_MARKER).touch(exist_ok=True)
            except Exception:
                candidate.close()
                raise
            self._client = candidate
        return callback(self._client)

    def _close_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _reset_storage(self) -> None:
        self._close_client()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        marker = self._path / _OWNER_MARKER
        if self._path.exists() and not marker.is_file():
            raise VectorStoreCorruptedError(
                "Le dossier Qdrant n'est pas marque comme index derive Second Brain. "
                "Reconstruction automatique refusee pour proteger les fichiers utilisateur."
            )
        quarantine = self._path.with_name(f".{self._path.name}.reset-{uuid4().hex}")
        moved = False
        if self._path.exists():
            self._path.replace(quarantine)
            moved = True
        try:
            candidate = QdrantClient(path=str(self._path))
            try:
                candidate.get_collections()
                (self._path / _OWNER_MARKER).touch(exist_ok=True)
            finally:
                candidate.close()
        except Exception:
            _remove_exact_path(self._path)
            if moved:
                quarantine.replace(self._path)
            raise
        if moved:
            try:
                _remove_exact_path(quarantine)
            except OSError:
                logger.warning("Qdrant reset quarantine cleanup failed")

    def _assert_adoptable_path(self) -> None:
        if not self._path.exists():
            return
        if not self._path.is_dir():
            raise VectorStoreCorruptedError(
                "QDRANT_PATH n'est pas un dossier d'index vectoriel valide et doit etre "
                "reconstruit. SQLite reste intacte."
            )
        entries = {entry.name for entry in self._path.iterdir()}
        if not entries or _OWNER_MARKER in entries:
            return
        if "meta.json" in entries and "collection" in entries:
            return
        raise VectorStoreCorruptedError(
            "QDRANT_PATH contient des fichiers qui ne sont pas un index Second Brain. "
            "SQLite reste intacte."
        )


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


def _stored_vector(
    point_id: models.ExtendedPointId,
    payload: object,
    raw_vector: object,
) -> StoredVector:
    knowledge_node_id, source_id, fingerprint = _validated_payload(point_id, payload)
    if not isinstance(raw_vector, list):
        raise VectorStoreCorruptedError(
            "Un point Qdrant ne contient pas de vecteur dense exploitable."
        )
    try:
        vector = tuple(_validated_vector(raw_vector))
    except ValueError as error:
        raise VectorStoreCorruptedError("Un point Qdrant contient un vecteur invalide.") from error
    return StoredVector(
        knowledge_node_id=knowledge_node_id,
        source_id=source_id,
        fingerprint=fingerprint,
        vector=vector,
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
    if isinstance(error, (json.JSONDecodeError, pickle.UnpicklingError, sqlite3.DatabaseError)):
        return VectorStoreCorruptedError(
            "L'index Qdrant local est illisible et doit etre reconstruit."
        )
    normalized = str(error).casefold()
    corruption_markers = (
        "corrupt",
        "deserialize",
        "invalid load key",
        "malformed",
        "unpickl",
        "not a directory",
        "is not a directory",
        "file exists",
        "cannot create a file when that file already exists",
    )
    if any(marker in normalized for marker in corruption_markers):
        return VectorStoreCorruptedError(
            "L'index Qdrant local est illisible et doit etre reconstruit."
        )
    return VectorStoreUnavailableError(
        "L'index Qdrant local est indisponible. Les donnees SQLite restent intactes."
    )


def _remove_exact_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
