from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from datetime import timedelta
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.migrations import migrate_database
from second_brain.db.models.embedding import (
    EmbeddingProfile,
    EmbeddingProfileStatus,
    KnowledgeEmbedding,
    KnowledgeEmbeddingStatus,
)
from second_brain.db.models.knowledge import KnowledgeNode
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.models.source import Source, SourceType
from second_brain.db.session import Database
from second_brain.llm.schemas import OllamaReadiness
from second_brain.main import create_app
from second_brain.services.vector_index import (
    VectorIndexIncompatibleError,
    VectorIndexService,
)
from second_brain.vector.embeddings import (
    EmbeddingBatchResult,
    EmbeddingCallContext,
    EmbeddingCallMetrics,
)
from second_brain.vector.store import (
    StoredVector,
    StoredVectorPoint,
    VectorCollectionInfo,
    VectorPoint,
    VectorSearchHit,
    VectorStoreUnavailableError,
)
from sqlalchemy import func, select


class FakeEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "qwen3-embedding:0.6b",
        fail_on_call: int | None = None,
    ) -> None:
        self.model = model
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    @property
    def configured_model(self) -> str:
        return self.model

    async def get_readiness(self) -> OllamaReadiness:
        return OllamaReadiness(
            ollama_available=True,
            configured_model=self.model,
            model_available=True,
            available_models=[self.model],
            error_code=None,
            message="Modele disponible.",
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        context: EmbeddingCallContext | None = None,
        metrics_callback=None,
    ) -> EmbeddingBatchResult:
        del context
        batch = list(texts)
        self.calls.append(batch)
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("panne simulee sans contenu prive")
        vectors = tuple(_fake_vector(text) for text in batch)
        metrics = EmbeddingCallMetrics(
            model=self.model,
            batch_size=len(batch),
            duration_seconds=0.01,
            total_duration_ns=10_000_000,
            load_duration_ns=0,
            prompt_eval_count=sum(len(text.split()) for text in batch),
            outcome="success",
        )
        if metrics_callback:
            metrics_callback(metrics)
        return EmbeddingBatchResult(
            model=self.model,
            vectors=vectors,
            dimension=3,
            metrics=metrics,
        )


class FakeVectorStore:
    def __init__(self) -> None:
        self.dimensions: dict[str, int] = {}
        self.points: dict[str, dict[UUID, VectorPoint]] = {}
        self.closed = False
        self.fail_upsert = False

    async def inspect_collection(self, collection_name: str) -> VectorCollectionInfo | None:
        dimension = self.dimensions.get(collection_name)
        return VectorCollectionInfo(collection_name, dimension) if dimension is not None else None

    async def ensure_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> VectorCollectionInfo:
        existing = self.dimensions.get(collection_name)
        if existing is not None and existing != dimension:
            raise AssertionError("dimension incompatible")
        self.dimensions[collection_name] = dimension
        self.points.setdefault(collection_name, {})
        return VectorCollectionInfo(collection_name, dimension)

    async def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None:
        if self.fail_upsert:
            raise VectorStoreUnavailableError("Qdrant simule indisponible.")
        self.points.setdefault(collection_name, {}).update(
            {point.knowledge_node_id: point for point in points}
        )

    async def retrieve(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> list[StoredVectorPoint]:
        collection = self.points.get(collection_name, {})
        return [
            StoredVectorPoint(point.knowledge_node_id, point.source_id, point.fingerprint)
            for node_id in knowledge_node_ids
            if (point := collection.get(node_id)) is not None
        ]

    async def retrieve_vectors(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> list[StoredVector]:
        collection = self.points.get(collection_name, {})
        return [
            StoredVector(
                point.knowledge_node_id,
                point.source_id,
                point.fingerprint,
                tuple(float(value) for value in point.vector),
            )
            for node_id in knowledge_node_ids
            if (point := collection.get(node_id)) is not None
        ]

    async def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        *,
        limit: int,
    ) -> list[VectorSearchHit]:
        hits = [
            VectorSearchHit(
                point.knowledge_node_id,
                point.source_id,
                point.fingerprint,
                _cosine(query_vector, point.vector),
            )
            for point in self.points.get(collection_name, {}).values()
        ]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    async def delete(
        self,
        collection_name: str,
        knowledge_node_ids: Sequence[UUID],
    ) -> None:
        for node_id in knowledge_node_ids:
            self.points.get(collection_name, {}).pop(node_id, None)

    async def list_point_ids(self, collection_name: str) -> set[UUID]:
        return set(self.points.get(collection_name, {}))

    async def delete_collection(self, collection_name: str) -> None:
        self.dimensions.pop(collection_name, None)
        self.points.pop(collection_name, None)

    async def close(self) -> None:
        self.closed = True


def _fake_vector(text: str) -> tuple[float, float, float]:
    normalized = text.casefold()
    if "plastique" in normalized or "pare-chocs" in normalized:
        return (1.0, 0.0, 0.0)
    if "recuperation" in normalized or "epuisement" in normalized:
        return (0.0, 1.0, 0.0)
    return (0.0, 0.0, 1.0)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


async def _database(settings: Settings) -> Database:
    settings.create_data_directory()
    await migrate_database(settings.resolved_database_url)
    return Database(settings.resolved_database_url)


async def _seed_nodes(database: Database, count: int) -> list[UUID]:
    async with database.session_factory() as session:
        source = Source(
            type=SourceType.MANUAL,
            title="Source de test",
            raw_text="Texte source conserve.",
        )
        session.add(source)
        await session.flush()
        nodes = [
            KnowledgeNode(
                source_id=source.id,
                title=f"Connaissance {index}",
                content=f"Contenu autonome numero {index} sur la recuperation.",
            )
            for index in range(count)
        ]
        session.add_all(nodes)
        await session.commit()
        return [node.id for node in nodes]


async def _finish_job(database: Database, job_id: UUID, status: ProcessingJobStatus) -> None:
    async with database.session_factory() as session:
        job = await session.get(ProcessingJob, job_id)
        assert job is not None
        job.status = status
        job.finished_at = utc_now()
        await session.commit()


@pytest.mark.anyio
async def test_indexing_batches_is_idempotent_and_observes_dimension(settings: Settings) -> None:
    database = await _database(settings)
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    try:
        await _seed_nodes(database, 17)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings.model_copy(update={"embedding_batch_size": 8}),
        )
        first_job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(first_job.id)
        await _finish_job(database, first_job.id, ProcessingJobStatus.SUCCEEDED)

        assert [len(batch) for batch in provider.calls] == [8, 8, 1]
        snapshot = await service.status()
        assert snapshot.state == "ready"
        assert snapshot.indexed_nodes == 17
        assert snapshot.profile is not None
        assert snapshot.profile.dimensions == 3
        assert len(store.points[snapshot.profile.collection_name]) == 17

        second_job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(second_job.id)
        await _finish_job(database, second_job.id, ProcessingJobStatus.SUCCEEDED)
        assert [len(batch) for batch in provider.calls] == [8, 8, 1]
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_interrupted_index_resumes_without_reembedding_completed_batch(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = FakeEmbeddingProvider(fail_on_call=2)
    store = FakeVectorStore()
    try:
        await _seed_nodes(database, 17)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings.model_copy(update={"embedding_batch_size": 8}),
        )
        failed_job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        with pytest.raises(RuntimeError, match="panne simulee"):
            await service.run_job(failed_job.id)
        await _finish_job(database, failed_job.id, ProcessingJobStatus.FAILED)

        assert [len(batch) for batch in provider.calls] == [8, 8]
        provider.fail_on_call = None
        resumed_job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(resumed_job.id)
        await _finish_job(database, resumed_job.id, ProcessingJobStatus.SUCCEEDED)

        assert [len(batch) for batch in provider.calls] == [8, 8, 8, 1]
        snapshot = await service.status()
        assert snapshot.indexed_nodes == 17
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_qdrant_point_without_sqlite_checkpoint_is_recovered_without_embedding(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    try:
        node_ids = await _seed_nodes(database, 2)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(job.id)
        await _finish_job(database, job.id, ProcessingJobStatus.SUCCEEDED)
        original_call_count = len(provider.calls)

        async with database.session_factory() as session:
            profile = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert profile is not None
            record = await session.get(KnowledgeEmbedding, (node_ids[0], profile.id))
            assert record is not None
            await session.delete(record)
            await session.commit()

        resumed = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(resumed.id)
        await _finish_job(database, resumed.id, ProcessingJobStatus.SUCCEEDED)
        assert len(provider.calls) == original_call_count
        async with database.session_factory() as session:
            recovered = await session.get(
                KnowledgeEmbedding,
                (node_ids[0], profile.id),
            )
            assert recovered is not None
            assert recovered.status == KnowledgeEmbeddingStatus.INDEXED
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_changed_content_is_reembedded_and_deleted_node_is_swept(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    try:
        node_ids = await _seed_nodes(database, 2)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        first = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(first.id)
        await _finish_job(database, first.id, ProcessingJobStatus.SUCCEEDED)
        call_count = len(provider.calls)

        async with database.session_factory() as session:
            node = await session.get(KnowledgeNode, node_ids[0])
            assert node is not None
            node.content = "Contenu modifie sur le plastique et le pare-chocs."
            deleted = await session.get(KnowledgeNode, node_ids[1])
            assert deleted is not None
            await session.delete(deleted)
            await session.commit()

        assert (await service.status()).state == "stale"
        second = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(second.id)
        await _finish_job(database, second.id, ProcessingJobStatus.SUCCEEDED)
        assert len(provider.calls) == call_count + 1
        snapshot = await service.status()
        assert snapshot.profile is not None
        assert set(store.points[snapshot.profile.collection_name]) == {node_ids[0]}
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_model_change_requires_rebuild_and_preserves_old_profile_on_failure(
    settings: Settings,
) -> None:
    database = await _database(settings)
    original_provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    try:
        await _seed_nodes(database, 2)
        original = VectorIndexService(
            database=database,
            embedding_provider=original_provider,
            vector_store=store,
            settings=settings,
        )
        first = await original.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await original.run_job(first.id)
        await _finish_job(database, first.id, ProcessingJobStatus.SUCCEEDED)

        changed_provider = FakeEmbeddingProvider(model="autre-embedding:1", fail_on_call=1)
        changed = VectorIndexService(
            database=database,
            embedding_provider=changed_provider,
            vector_store=store,
            settings=settings,
        )
        with pytest.raises(VectorIndexIncompatibleError):
            await changed.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)

        rebuild = await changed.prepare_job(ProcessingJobKind.REBUILD_VECTOR_INDEX)
        with pytest.raises(RuntimeError):
            await changed.run_job(rebuild.id)
        async with database.session_factory() as session:
            active = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert active is not None
            assert active.model_name == "qwen3-embedding:0.6b"
            assert int((await session.scalar(select(func.count(KnowledgeNode.id)))) or 0) == 2
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_semantic_search_reloads_ranked_nodes_from_sqlite(settings: Settings) -> None:
    database = await _database(settings)
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    try:
        async with database.session_factory() as session:
            source = Source(type=SourceType.MANUAL, title="Atelier", raw_text="Original")
            session.add(source)
            await session.flush()
            plastic = KnowledgeNode(
                source_id=source.id,
                title="Preparation des polymeres",
                content="Un pare-chocs propre recoit mieux la peinture.",
            )
            recovery = KnowledgeNode(
                source_id=source.id,
                title="Gestion de la fatigue",
                content="La recuperation evite l'epuisement entre les seances.",
            )
            session.add_all([plastic, recovery])
            await session.commit()

        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        await service.run_job(job.id)
        await _finish_job(database, job.id, ProcessingJobStatus.SUCCEEDED)

        results = await service.search(
            "Comment traiter un pare-chocs en polymere avant peinture ?",
            top_k=2,
        )
        assert [item.node.id for item in results.items] == [plastic.id, recovery.id]
        assert results.items[0].node.source.title == "Atelier"
        assert results.items[0].score > results.items[1].score
        assert results.timings.embedding_seconds >= 0
        assert results.timings.qdrant_seconds >= 0
        assert results.timings.sqlite_seconds >= 0
        assert results.timings.total_seconds >= results.timings.embedding_seconds
    finally:
        await database.dispose()


def test_vector_api_exposes_empty_search_and_persistent_job(
    settings: Settings,
) -> None:
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    app = create_app(
        settings,
        embedding_provider=provider,
        vector_store=store,
        start_analysis_worker=False,
        start_indexing_worker=False,
    )
    with TestClient(app) as client:
        status_response = client.get("/api/v1/vector-index/status")
        assert status_response.status_code == 200
        assert status_response.json()["state"] == "empty"
        assert status_response.json()["configured_model"] == "qwen3-embedding:0.6b"

        search = client.post(
            "/api/v1/search/semantic",
            json={"query": "question semantique", "top_k": 5},
        )
        assert search.status_code == 200
        assert search.json()["items"] == []

        queued = client.post("/api/v1/vector-index/index")
        assert queued.status_code == 202
        job_id = queued.json()["id"]
        assert queued.json()["kind"] == "index_knowledge"
        assert queued.json()["progress_total"] == 0
        assert client.get(f"/api/v1/vector-index/jobs/{job_id}").status_code == 200

        duplicate = client.post("/api/v1/vector-index/index")
        assert duplicate.status_code == 202
        assert duplicate.json()["id"] == job_id

        unconfirmed = client.post("/api/v1/vector-index/rebuild", json={"confirm": False})
        assert unconfirmed.status_code == 422

    assert store.closed is True


def test_vector_api_indexes_searches_and_rebuilds_complete_sqlite_nodes(
    settings: Settings,
) -> None:
    async def seed() -> list[UUID]:
        database = await _database(settings)
        try:
            return await _seed_nodes(database, 2)
        finally:
            await database.dispose()

    node_ids = asyncio.run(seed())
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    app = create_app(
        settings,
        embedding_provider=provider,
        vector_store=store,
        start_analysis_worker=False,
        start_indexing_worker=True,
    )

    with TestClient(app) as client:
        queued = client.post("/api/v1/vector-index/index")
        assert queued.status_code == 202
        first_job = _wait_http_vector_job(client, queued.json()["id"])
        assert first_job["status"] == "succeeded"
        assert first_job["progress_current"] == len(node_ids)

        status_response = client.get("/api/v1/vector-index/status")
        assert status_response.status_code == 200
        first_status = status_response.json()
        assert first_status["state"] == "ready"
        assert first_status["indexed_nodes"] == len(node_ids)
        assert first_status["active_profile"]["dimensions"] == 3

        search = client.post(
            "/api/v1/search/semantic",
            json={"query": "Comment favoriser la recuperation ?", "top_k": 2},
        )
        assert search.status_code == 200
        search_payload = search.json()
        assert len(search_payload["items"]) == 2
        assert {item["knowledge_node"]["id"] for item in search_payload["items"]} == {
            str(node_id) for node_id in node_ids
        }
        assert all(item["source"]["title"] == "Source de test" for item in search_payload["items"])

        rebuilt = client.post(
            "/api/v1/vector-index/rebuild",
            json={"confirm": True},
        )
        assert rebuilt.status_code == 202
        rebuilt_job = _wait_http_vector_job(client, rebuilt.json()["id"])
        assert rebuilt_job["status"] == "succeeded"

        final_status = client.get("/api/v1/vector-index/status").json()
        assert final_status["state"] == "ready"
        assert final_status["indexed_nodes"] == len(node_ids)
        assert final_status["active_profile"]["logical_generation"] == 2

    assert store.closed is True


def _wait_http_vector_job(
    client: TestClient,
    job_id: str,
    *,
    timeout_seconds: float = 3,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/vector-index/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Le job vectoriel {job_id} n'a pas termine.")


@pytest.mark.anyio
async def test_stale_vector_job_is_reported_by_status(settings: Settings) -> None:
    database = await _database(settings)
    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    try:
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        async with database.session_factory() as session:
            persisted = await session.get(ProcessingJob, job.id)
            assert persisted is not None
            persisted.status = ProcessingJobStatus.RUNNING
            persisted.last_activity_at = utc_now() - timedelta(hours=1)
            await session.commit()
        snapshot = await service.status()
        assert snapshot.active_job is not None
        assert snapshot.active_job.id == job.id
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_concurrent_index_requests_create_only_one_active_job(settings: Settings) -> None:
    database = await _database(settings)
    try:
        service = VectorIndexService(
            database=database,
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=FakeVectorStore(),
            settings=settings,
        )
        jobs = await asyncio.gather(
            *(service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE) for _ in range(5))
        )
        assert len({job.id for job in jobs}) == 1
        async with database.session_factory() as session:
            count = int((await session.scalar(select(func.count(ProcessingJob.id)))) or 0)
            assert count == 1
    finally:
        await database.dispose()
