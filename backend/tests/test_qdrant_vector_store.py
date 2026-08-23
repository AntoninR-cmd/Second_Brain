from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import QdrantClient, models
from second_brain.vector.qdrant_store import QdrantVectorStore
from second_brain.vector.store import (
    VectorPoint,
    VectorStoreCompatibilityError,
    VectorStoreCorruptedError,
    VectorStoreUnavailableError,
)


@pytest.mark.anyio
async def test_qdrant_local_store_persists_searches_and_upserts_without_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qdrant"
    collection = "second_brain_test"
    first_node_id = uuid4()
    second_node_id = uuid4()
    first_source_id = uuid4()
    second_source_id = uuid4()
    store = QdrantVectorStore(path)

    try:
        info = await store.ensure_collection(collection, 3)
        assert info.name == collection
        assert info.dimension == 3
        assert info.distance == "cosine"
        assert await store.inspect_collection(collection) == info

        await store.upsert(
            collection,
            [
                VectorPoint(first_node_id, first_source_id, "a" * 64, [1.0, 0.0, 0.0]),
                VectorPoint(second_node_id, second_source_id, "b" * 64, [0.0, 1.0, 0.0]),
            ],
        )
        assert await store.list_point_ids(collection) == {first_node_id, second_node_id}
        assert [
            point.knowledge_node_id
            for point in await store.retrieve(collection, [second_node_id, first_node_id])
        ] == [second_node_id, first_node_id]

        hits = await store.search(collection, [0.9, 0.1, 0.0], limit=2)
        assert [hit.knowledge_node_id for hit in hits] == [first_node_id, second_node_id]
        assert hits[0].score > hits[1].score

        await store.upsert(
            collection,
            [VectorPoint(first_node_id, first_source_id, "c" * 64, [1.0, 0.0, 0.0])],
        )
        assert await store.list_point_ids(collection) == {first_node_id, second_node_id}
        updated = await store.retrieve(collection, [first_node_id])
        assert updated[0].fingerprint == "c" * 64

        await store.delete(collection, [second_node_id])
        assert await store.list_point_ids(collection) == {first_node_id}
    finally:
        await store.close()

    reopened = QdrantVectorStore(path)
    try:
        assert await reopened.inspect_collection(collection) == info
        assert await reopened.list_point_ids(collection) == {first_node_id}
    finally:
        await reopened.close()

    raw_client = QdrantClient(path=str(path))
    try:
        raw_point = raw_client.retrieve(
            collection_name=collection,
            ids=[str(first_node_id)],
            with_payload=True,
            with_vectors=False,
        )[0]
        assert raw_point.payload == {
            "knowledge_node_id": str(first_node_id),
            "source_id": str(first_source_id),
            "fingerprint": "c" * 64,
        }
    finally:
        raw_client.close()


@pytest.mark.anyio
async def test_qdrant_collection_rejects_incompatible_dimension_and_distance(
    tmp_path: Path,
) -> None:
    dimension_path = tmp_path / "dimension"
    store = QdrantVectorStore(dimension_path)
    try:
        await store.ensure_collection("nodes", 3)
        with pytest.raises(VectorStoreCompatibilityError):
            await store.ensure_collection("nodes", 4)
    finally:
        await store.close()

    distance_path = tmp_path / "distance"
    raw_client = QdrantClient(path=str(distance_path))
    try:
        raw_client.create_collection(
            collection_name="nodes",
            vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT),
        )
    finally:
        raw_client.close()

    incompatible = QdrantVectorStore(distance_path)
    try:
        with pytest.raises(VectorStoreCompatibilityError):
            await incompatible.inspect_collection("nodes")
    finally:
        await incompatible.close()


@pytest.mark.anyio
async def test_qdrant_store_detects_corrupt_payload_and_non_uuid_point(
    tmp_path: Path,
) -> None:
    path = tmp_path / "corrupt-payload"
    raw_client = QdrantClient(path=str(path))
    try:
        raw_client.create_collection(
            collection_name="nodes",
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        raw_client.upsert(
            collection_name="nodes",
            points=[models.PointStruct(id=7, vector=[1.0, 0.0], payload={"unexpected": True})],
        )
    finally:
        raw_client.close()

    store = QdrantVectorStore(path)
    try:
        with pytest.raises(VectorStoreCorruptedError):
            await store.list_point_ids("nodes")
    finally:
        await store.close()


@pytest.mark.anyio
async def test_qdrant_store_validates_vectors_and_releases_resources(tmp_path: Path) -> None:
    store = QdrantVectorStore(tmp_path / "qdrant")
    node_id = uuid4()
    source_id = uuid4()
    try:
        await store.ensure_collection("nodes", 2)
        with pytest.raises(ValueError):
            await store.upsert(
                "nodes",
                [VectorPoint(node_id, source_id, "a" * 64, [1.0, float("nan")])],
            )
        with pytest.raises(ValueError):
            await store.search("nodes", [], limit=5)
        with pytest.raises(ValueError):
            await store.search("nodes", [1.0, 0.0], limit=0)
    finally:
        await store.close()

    await store.close()
    with pytest.raises(VectorStoreUnavailableError):
        await store.inspect_collection("nodes")


@pytest.mark.anyio
async def test_invalid_qdrant_path_is_mapped_without_exposing_sqlite(tmp_path: Path) -> None:
    invalid_path = tmp_path / "not-a-directory"
    invalid_path.write_text("index illisible", encoding="utf-8")
    store = QdrantVectorStore(invalid_path)
    try:
        with pytest.raises((VectorStoreUnavailableError, VectorStoreCorruptedError)) as raised:
            await store.inspect_collection("nodes")
        assert "SQLite" in str(raised.value) or "reconstruit" in str(raised.value)
    finally:
        await store.close()
