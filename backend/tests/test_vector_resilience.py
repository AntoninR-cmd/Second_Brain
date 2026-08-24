from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import httpx2
import pytest
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
from second_brain.db.models.taxonomy import KnowledgeNodeTag, Tag
from second_brain.db.session import Database
from second_brain.jobs.indexing_runner import IndexingRunner
from second_brain.llm.errors import OllamaUnavailableError
from second_brain.llm.schemas import OllamaReadiness
from second_brain.services.vector_index import (
    VectorIndexIncompatibleError,
    VectorIndexIncompleteError,
    VectorIndexService,
)
from second_brain.vector.embeddings import (
    EmbeddingBatchResult,
    EmbeddingCallContext,
    EmbeddingCallMetrics,
    OllamaEmbeddingProvider,
)
from second_brain.vector.qdrant_store import QdrantVectorStore
from second_brain.vector.store import (
    StoredVector,
    StoredVectorPoint,
    VectorCollectionInfo,
    VectorPoint,
    VectorSearchHit,
    VectorStoreCompatibilityError,
    VectorStoreCorruptedError,
    VectorStoreUnavailableError,
)
from sqlalchemy import func, select


class ControlledEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "qwen3-embedding:0.6b",
        digest: str | None = "sha256:model-a",
        dimension: int = 3,
        fail_on_call: int | None = None,
        available: bool = True,
    ) -> None:
        self.model = model
        self.digest = digest
        self.dimension = dimension
        self.fail_on_call = fail_on_call
        self.available = available
        self.calls: list[list[str]] = []

    @property
    def configured_model(self) -> str:
        return self.model

    async def get_readiness(self) -> OllamaReadiness:
        return OllamaReadiness(
            ollama_available=self.available,
            configured_model=self.model,
            configured_model_digest=self.digest if self.available else None,
            model_available=self.available,
            available_models=[self.model] if self.available else [],
            error_code=None if self.available else "unavailable",
            message=(
                "Modele d'embedding disponible."
                if self.available
                else "Ollama est indisponible pour le test."
            ),
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
        if not self.available or self.fail_on_call == len(self.calls):
            raise OllamaUnavailableError("Ollama indisponible pendant l'indexation.")

        vectors = tuple(_vector_for(text, self.dimension) for text in batch)
        metrics = EmbeddingCallMetrics(
            model=self.model,
            batch_size=len(batch),
            duration_seconds=0.01,
            total_duration_ns=10_000_000,
            load_duration_ns=0,
            prompt_eval_count=sum(len(text.split()) for text in batch),
            outcome="success",
        )
        if metrics_callback is not None:
            metrics_callback(metrics)
        return EmbeddingBatchResult(
            model=self.model,
            vectors=vectors,
            dimension=self.dimension,
            metrics=metrics,
        )


class ControlledVectorStore:
    def __init__(self) -> None:
        self.dimensions: dict[str, int] = {}
        self.points: dict[str, dict[UUID, VectorPoint]] = {}
        self.inspect_error: Exception | None = None
        self.upsert_error: Exception | None = None
        self.deleted_collections: list[str] = []
        self.drop_last_upsert_point = False
        self.reset_count = 0

    async def inspect_collection(self, collection_name: str) -> VectorCollectionInfo | None:
        if self.inspect_error is not None:
            raise self.inspect_error
        dimension = self.dimensions.get(collection_name)
        if dimension is None:
            return None
        return VectorCollectionInfo(collection_name, dimension)

    async def ensure_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> VectorCollectionInfo:
        existing = self.dimensions.get(collection_name)
        if existing is not None and existing != dimension:
            raise VectorStoreCompatibilityError("Dimension Qdrant incompatible.")
        self.dimensions[collection_name] = dimension
        self.points.setdefault(collection_name, {})
        return VectorCollectionInfo(collection_name, dimension)

    async def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None:
        if self.upsert_error is not None:
            raise self.upsert_error
        persisted_points = list(points)
        if self.drop_last_upsert_point:
            persisted_points = persisted_points[:-1]
        self.points.setdefault(collection_name, {}).update(
            {point.knowledge_node_id: point for point in persisted_points}
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
        del query_vector
        return [
            VectorSearchHit(point.knowledge_node_id, point.source_id, point.fingerprint, 1.0)
            for point in list(self.points.get(collection_name, {}).values())[:limit]
        ]

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
        self.deleted_collections.append(collection_name)
        self.dimensions.pop(collection_name, None)
        self.points.pop(collection_name, None)

    async def reset_storage(self) -> None:
        self.reset_count += 1
        self.inspect_error = None
        self.dimensions.clear()
        self.points.clear()

    async def close(self) -> None:
        return None


def _vector_for(text: str, dimension: int) -> tuple[float, ...]:
    seed = sum(text.encode("utf-8")) or 1
    return tuple(float((seed + index) % 251 + 1) for index in range(dimension))


async def _database(settings: Settings) -> Database:
    settings.create_data_directory()
    await migrate_database(settings.resolved_database_url)
    return Database(settings.resolved_database_url)


async def _seed_nodes(database: Database, count: int) -> list[UUID]:
    async with database.session_factory() as session:
        source = Source(
            type=SourceType.MANUAL,
            title="Source resiliente",
            raw_text="Texte source prive qui doit rester dans SQLite.",
        )
        session.add(source)
        await session.flush()
        nodes = [
            KnowledgeNode(
                source_id=source.id,
                title=f"Connaissance {index}",
                content=f"Contenu semantique autonome {index}.",
            )
            for index in range(count)
        ]
        session.add_all(nodes)
        await session.commit()
        return [node.id for node in nodes]


async def _set_job_status(
    database: Database,
    job_id: UUID,
    status: ProcessingJobStatus,
) -> None:
    async with database.session_factory() as session:
        job = await session.get(ProcessingJob, job_id)
        assert job is not None
        job.status = status
        job.finished_at = (
            utc_now()
            if status
            in {
                ProcessingJobStatus.SUCCEEDED,
                ProcessingJobStatus.FAILED,
            }
            else None
        )
        await session.commit()


async def _run_successfully(
    service: VectorIndexService,
    database: Database,
    kind: ProcessingJobKind = ProcessingJobKind.INDEX_KNOWLEDGE,
) -> ProcessingJob:
    job = await service.prepare_job(kind)
    await service.run_job(job.id)
    await _set_job_status(database, job.id, ProcessingJobStatus.SUCCEEDED)
    return job


async def _wait_for_job(
    database: Database,
    job_id: UUID,
    expected: ProcessingJobStatus,
    *,
    timeout_seconds: float = 3,
) -> ProcessingJob:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        async with database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is not None and job.status == expected:
                return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"Le job {job_id} n'a pas atteint l'etat {expected.value}.")


@pytest.mark.anyio
async def test_ollama_readiness_preserves_the_configured_model_digest(
    settings: Settings,
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/tags"
        return httpx2.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3-embedding:0.6b",
                        "digest": "sha256:immutable-revision",
                    },
                    {"name": "qwen3.5:4b", "digest": "sha256:generation-model"},
                ]
            },
        )

    provider = OllamaEmbeddingProvider(
        settings,
        transport=httpx2.MockTransport(handler),
    )
    readiness = await provider.get_readiness()

    assert readiness.model_available is True
    assert readiness.configured_model == "qwen3-embedding:0.6b"
    assert readiness.configured_model_digest == "sha256:immutable-revision"


@pytest.mark.anyio
async def test_same_model_tag_with_a_new_digest_requires_a_successful_rebuild(
    settings: Settings,
) -> None:
    database = await _database(settings)
    store = ControlledVectorStore()
    first_provider = ControlledEmbeddingProvider(digest="sha256:revision-a")
    try:
        await _seed_nodes(database, 2)
        first_service = VectorIndexService(
            database=database,
            embedding_provider=first_provider,
            vector_store=store,
            settings=settings,
        )
        await _run_successfully(first_service, database)

        async with database.session_factory() as session:
            old_profile = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert old_profile is not None
            old_profile_id = old_profile.id
            assert old_profile.model_digest == "sha256:revision-a"

        changed_service = VectorIndexService(
            database=database,
            embedding_provider=ControlledEmbeddingProvider(digest="sha256:revision-b"),
            vector_store=store,
            settings=settings,
        )
        assert (await changed_service.status()).state == "incompatible"
        with pytest.raises(VectorIndexIncompatibleError):
            await changed_service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)

        await _run_successfully(
            changed_service,
            database,
            ProcessingJobKind.REBUILD_VECTOR_INDEX,
        )
        snapshot = await changed_service.status()
        assert snapshot.state == "ready"
        assert snapshot.profile is not None
        assert snapshot.profile.id != old_profile_id
        assert snapshot.profile.model_digest == "sha256:revision-b"

        async with database.session_factory() as session:
            old_profile = await session.get(EmbeddingProfile, old_profile_id)
            assert old_profile is not None
            assert old_profile.status == EmbeddingProfileStatus.RETIRED
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_semantic_changes_persist_stale_but_tag_changes_do_not(
    settings: Settings,
) -> None:
    database = await _database(settings)
    store = ControlledVectorStore()
    provider = ControlledEmbeddingProvider()
    try:
        node_ids = await _seed_nodes(database, 3)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        await _run_successfully(service, database)

        async with database.session_factory() as session:
            title_changed = await session.get(KnowledgeNode, node_ids[0])
            content_changed = await session.get(KnowledgeNode, node_ids[1])
            assert title_changed is not None
            assert content_changed is not None
            title_changed.title = "Titre semantique modifie"
            content_changed.content = "Contenu semantique modifie."

            tag = Tag(name="Nouveau tag", normalized_name="nouveau-tag")
            session.add(tag)
            await session.flush()
            session.add(KnowledgeNodeTag(knowledge_node_id=node_ids[2], tag_id=tag.id))
            await session.commit()

        async with database.session_factory() as session:
            profile = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert profile is not None
            statuses = {
                node_id: (await session.get(KnowledgeEmbedding, (node_id, profile.id))).status
                for node_id in node_ids
            }

        assert statuses == {
            node_ids[0]: KnowledgeEmbeddingStatus.STALE,
            node_ids[1]: KnowledgeEmbeddingStatus.STALE,
            node_ids[2]: KnowledgeEmbeddingStatus.INDEXED,
        }
        snapshot = await service.status()
        assert snapshot.state == "stale"
        assert snapshot.indexed_nodes == 1
        assert snapshot.pending_or_stale_nodes == 2
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_failed_first_index_keeps_diagnostic_and_sqlite_data_after_reload(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider(fail_on_call=1)
    store = ControlledVectorStore()
    runner: IndexingRunner | None = None
    try:
        node_ids = await _seed_nodes(database, 2)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        runner = IndexingRunner(database=database, service=service, settings=settings)
        await runner.start()
        job = await runner.enqueue(ProcessingJobKind.INDEX_KNOWLEDGE)
        failed = await _wait_for_job(database, job.id, ProcessingJobStatus.FAILED)
        assert failed.error_code == "ollama_unavailable"
        assert "Ollama indisponible" in (failed.error_detail or "")
        await runner.stop()
        runner = None
        await database.dispose()

        reloaded_database = Database(settings.resolved_database_url)
        reloaded_service = VectorIndexService(
            database=reloaded_database,
            embedding_provider=ControlledEmbeddingProvider(),
            vector_store=store,
            settings=settings,
        )
        try:
            snapshot = await reloaded_service.status()
            assert snapshot.active_job is None
            assert snapshot.error is not None
            assert "Ollama indisponible" in snapshot.error
            async with reloaded_database.session_factory() as session:
                assert int((await session.scalar(select(func.count(KnowledgeNode.id)))) or 0) == 2
                assert {
                    node.id for node in (await session.scalars(select(KnowledgeNode))).all()
                } == set(node_ids)
                persisted = await session.get(ProcessingJob, job.id)
                assert persisted is not None
                assert persisted.status == ProcessingJobStatus.FAILED
                assert persisted.error_code == "ollama_unavailable"
        finally:
            await reloaded_database.dispose()
    finally:
        if runner is not None:
            await runner.stop()
        await database.dispose()


@pytest.mark.anyio
async def test_rebuild_after_all_nodes_are_deleted_cleans_derived_points_without_embedding(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider()
    store = ControlledVectorStore()
    try:
        node_ids = await _seed_nodes(database, 1)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        await _run_successfully(service, database)
        initial_call_count = len(provider.calls)

        async with database.session_factory() as session:
            node = await session.get(KnowledgeNode, node_ids[0])
            assert node is not None
            await session.delete(node)
            await session.commit()

        await _run_successfully(
            service,
            database,
            ProcessingJobKind.REBUILD_VECTOR_INDEX,
        )
        assert len(provider.calls) == initial_call_count
        assert all(not collection for collection in store.points.values())
        snapshot = await service.status()
        assert snapshot.state in {"empty", "ready"}
        assert snapshot.total_nodes == 0
        assert snapshot.indexed_nodes == 0
        assert snapshot.orphan_points == 0
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_runner_restart_recovers_qdrant_checkpoint_without_reembedding(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider()
    store = ControlledVectorStore()
    runner: IndexingRunner | None = None
    try:
        node_ids = await _seed_nodes(database, 2)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings.model_copy(update={"embedding_batch_size": 2}),
        )
        await _run_successfully(service, database)
        provider.calls.clear()

        async with database.session_factory() as session:
            profile = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert profile is not None
            profile_id = profile.id
            collection_name = profile.collection_name
            for node_id in node_ids:
                checkpoint = await session.get(KnowledgeEmbedding, (node_id, profile.id))
                assert checkpoint is not None
                await session.delete(checkpoint)
            store.points[collection_name].pop(node_ids[1])

            interrupted = ProcessingJob(
                source_id=None,
                embedding_profile_id=profile.id,
                kind=ProcessingJobKind.INDEX_KNOWLEDGE,
                status=ProcessingJobStatus.RUNNING,
                stage="indexing",
                progress_current=1,
                progress_total=2,
                progress_percent=50,
                progress_message="Indexation interrompue : 1 / 2.",
                started_at=utc_now(),
            )
            session.add(interrupted)
            await session.commit()
            interrupted_id = interrupted.id

        runner = IndexingRunner(database=database, service=service, settings=settings)
        await runner.start()
        succeeded = await _wait_for_job(
            database,
            interrupted_id,
            ProcessingJobStatus.SUCCEEDED,
        )
        assert succeeded.attempt_count == 1
        assert [len(batch) for batch in provider.calls] == [1]

        async with database.session_factory() as session:
            checkpoints = (
                await session.scalars(
                    select(KnowledgeEmbedding).where(
                        KnowledgeEmbedding.embedding_profile_id == profile_id
                    )
                )
            ).all()
            assert {record.knowledge_node_id for record in checkpoints} == set(node_ids)
            assert all(record.status == KnowledgeEmbeddingStatus.INDEXED for record in checkpoints)
    finally:
        if runner is not None:
            await runner.stop()
        await database.dispose()


@pytest.mark.anyio
@pytest.mark.parametrize("failure_kind", ["ollama", "qdrant"])
async def test_external_index_failure_never_changes_sqlite_knowledge(
    settings: Settings,
    failure_kind: str,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider(fail_on_call=1 if failure_kind == "ollama" else None)
    store = ControlledVectorStore()
    if failure_kind == "qdrant":
        store.upsert_error = VectorStoreUnavailableError("Qdrant indisponible pour le test.")
    try:
        node_ids = await _seed_nodes(database, 2)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        expected_error = (
            OllamaUnavailableError if failure_kind == "ollama" else VectorStoreUnavailableError
        )
        with pytest.raises(expected_error):
            await service.run_job(job.id)

        async with database.session_factory() as session:
            nodes = list(
                (
                    await session.scalars(select(KnowledgeNode).order_by(KnowledgeNode.id.asc()))
                ).all()
            )
            assert {node.id for node in nodes} == set(node_ids)
            assert {node.content for node in nodes} == {
                "Contenu semantique autonome 0.",
                "Contenu semantique autonome 1.",
            }
    finally:
        await database.dispose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("store_error", "expected_state"),
    [
        (VectorStoreUnavailableError("Qdrant verrouille."), "unavailable"),
        (VectorStoreCorruptedError("Qdrant corrompu."), "corrupt"),
    ],
)
async def test_qdrant_health_error_is_reported_without_touching_sqlite(
    settings: Settings,
    store_error: Exception,
    expected_state: str,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider()
    store = ControlledVectorStore()
    try:
        node_ids = await _seed_nodes(database, 1)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        await _run_successfully(service, database)
        store.inspect_error = store_error

        snapshot = await service.status()
        assert snapshot.state == expected_state
        assert snapshot.error is not None
        async with database.session_factory() as session:
            node = await session.get(KnowledgeNode, node_ids[0])
            assert node is not None
            assert node.content == "Contenu semantique autonome 0."
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_incomplete_index_exposes_checkpoints_and_resumes_only_missing_nodes(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider(fail_on_call=2)
    store = ControlledVectorStore()
    try:
        node_ids = await _seed_nodes(database, 3)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings.model_copy(update={"embedding_batch_size": 2}),
        )
        failed_job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        with pytest.raises(OllamaUnavailableError):
            await service.run_job(failed_job.id)
        await _set_job_status(database, failed_job.id, ProcessingJobStatus.FAILED)

        snapshot = await service.status()
        assert snapshot.indexed_nodes == 2
        assert snapshot.failed_nodes == 1
        assert snapshot.pending_or_stale_nodes == 0
        async with database.session_factory() as session:
            profile = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.BUILDING
                )
            )
            assert profile is not None
            records = (
                await session.scalars(
                    select(KnowledgeEmbedding).where(
                        KnowledgeEmbedding.embedding_profile_id == profile.id
                    )
                )
            ).all()
            assert sum(record.status == KnowledgeEmbeddingStatus.INDEXED for record in records) == 2
            assert sum(record.status == KnowledgeEmbeddingStatus.FAILED for record in records) == 1

        provider.fail_on_call = None
        await _run_successfully(service, database)
        assert [len(batch) for batch in provider.calls] == [2, 1, 1]
        final = await service.status()
        assert final.state == "ready"
        assert final.indexed_nodes == 3
        assert final.failed_nodes == 0
        assert set(store.points[final.profile.collection_name]) == set(node_ids)  # type: ignore[union-attr]
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_silent_partial_upsert_never_activates_an_incomplete_profile(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider()
    store = ControlledVectorStore()
    store.drop_last_upsert_point = True
    try:
        node_ids = await _seed_nodes(database, 3)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        job = await service.prepare_job(ProcessingJobKind.INDEX_KNOWLEDGE)
        with pytest.raises(VectorIndexIncompleteError, match="nombre de points"):
            await service.run_job(job.id)
        await _set_job_status(database, job.id, ProcessingJobStatus.FAILED)

        async with database.session_factory() as session:
            active = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert active is None
        snapshot = await service.status()
        assert snapshot.error is not None
        assert "incomplet" in snapshot.error

        store.drop_last_upsert_point = False
        provider.calls.clear()
        await _run_successfully(service, database)
        assert [len(call) for call in provider.calls] == [1]
        final = await service.status()
        assert final.state == "ready"
        assert final.indexed_nodes == len(node_ids)
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_confirmed_rebuild_recovers_a_corrupt_owned_qdrant_store(
    settings: Settings,
    tmp_path: Path,
) -> None:
    database = await _database(settings)
    qdrant_path = tmp_path / "vector-data" / "qdrant"
    provider = ControlledEmbeddingProvider()
    first_store = QdrantVectorStore(qdrant_path, reset_root=tmp_path)
    try:
        node_ids = await _seed_nodes(database, 2)
        first_service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=first_store,
            settings=settings,
        )
        await _run_successfully(first_service, database)
    finally:
        await first_store.close()

    (qdrant_path / "meta.json").write_text("{invalid", encoding="utf-8")
    recovered_store = QdrantVectorStore(qdrant_path, reset_root=tmp_path)
    try:
        recovered_service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=recovered_store,
            settings=settings,
        )
        await _run_successfully(
            recovered_service,
            database,
            ProcessingJobKind.REBUILD_VECTOR_INDEX,
        )
        snapshot = await recovered_service.status()
        assert snapshot.state == "ready"
        assert snapshot.indexed_nodes == len(node_ids)
        async with database.session_factory() as session:
            assert int((await session.scalar(select(func.count(KnowledgeNode.id)))) or 0) == len(
                node_ids
            )
    finally:
        await recovered_store.close()
        await database.dispose()


@pytest.mark.anyio
async def test_changed_embedding_dimension_is_rejected_until_rebuild(
    settings: Settings,
) -> None:
    database = await _database(settings)
    provider = ControlledEmbeddingProvider(dimension=3)
    store = ControlledVectorStore()
    try:
        await _seed_nodes(database, 2)
        service = VectorIndexService(
            database=database,
            embedding_provider=provider,
            vector_store=store,
            settings=settings,
        )
        await _run_successfully(service, database)

        provider.dimension = 4
        with pytest.raises(VectorIndexIncompatibleError, match="dimension differente"):
            await service.search("Question semantique differente", top_k=2)

        await _run_successfully(
            service,
            database,
            ProcessingJobKind.REBUILD_VECTOR_INDEX,
        )
        snapshot = await service.status()
        assert snapshot.state == "ready"
        assert snapshot.profile is not None
        assert snapshot.profile.dimensions == 4
    finally:
        await database.dispose()
