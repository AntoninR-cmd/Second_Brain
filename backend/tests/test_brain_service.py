from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

import pytest
from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.migrations import migrate_database
from second_brain.db.models.brain import (
    BrainCluster,
    BrainEdge,
    BrainLabelSource,
    BrainLabelStrategy,
    BrainNodeLayout,
    BrainProfile,
    BrainProfileStatus,
)
from second_brain.db.models.embedding import (
    EmbeddingDistance,
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
from second_brain.graph import (
    BrainCluster as MathCluster,
)
from second_brain.graph import (
    BrainEdge as MathEdge,
)
from second_brain.graph import (
    BrainMathDurations,
    BrainMathResult,
    BrainMathStats,
    SimilarityDistribution,
)
from second_brain.graph import (
    BrainNodeLayout as MathNodeLayout,
)
from second_brain.jobs.brain_runner import BrainRunner
from second_brain.llm.client import GenerationAttemptMetrics
from second_brain.llm.schemas import OllamaReadiness
from second_brain.services.brain import BrainService
from second_brain.services.vector_index import (
    ActiveEmbeddingCorpus,
    ActiveEmbeddingNode,
    ActiveEmbeddingProfile,
)
from second_brain.vector.semantic_text import semantic_text_fingerprint
from sqlalchemy import func, select


class FakeVectorService:
    def __init__(self, corpus: ActiveEmbeddingCorpus) -> None:
        self.corpus = corpus
        self.calls = 0

    async def load_active_corpus(self) -> ActiveEmbeddingCorpus:
        self.calls += 1
        return self.corpus


class FakeBrainGenerator:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls = 0

    async def get_readiness(self) -> OllamaReadiness:
        return OllamaReadiness(
            ollama_available=self.available,
            configured_model="qwen3.5:4b",
            configured_model_digest="generation-digest",
            model_available=self.available,
            available_models=["qwen3.5:4b"] if self.available else [],
            error_code=None if self.available else "unavailable",
            message=("Modele disponible." if self.available else "Ollama indisponible."),
        )

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type,
        call_type: str,
        system_prompt: str | None = None,
        context=None,
        metrics_callback: Callable[[GenerationAttemptMetrics], None] | None = None,
        result_validator: Callable[[object], None] | None = None,
    ):
        del prompt, system_prompt, context
        self.calls += 1
        assert call_type == "cluster_labeling"
        response = response_model.model_validate(
            {
                "labels": [
                    {
                        "cluster_key": "c0001",
                        "label": "Theme genere",
                        "description": "Description courte du theme.",
                    }
                ]
            }
        )
        if result_validator is not None:
            result_validator(response)
        if metrics_callback is not None:
            metrics_callback(
                GenerationAttemptMetrics(
                    call_type="cluster_labeling",
                    attempt=1,
                    duration_seconds=0.025,
                    total_duration_ns=25_000_000,
                    prompt_eval_count=42,
                    prompt_eval_duration_ns=4_000_000,
                    eval_count=8,
                    eval_duration_ns=18_000_000,
                    outcome="success",
                )
            )
        return response


@dataclass(frozen=True)
class SeededCorpus:
    corpus: ActiveEmbeddingCorpus
    node_ids: tuple[UUID, ...]
    source_id: UUID


async def _database(settings: Settings) -> Database:
    settings.create_data_directory()
    await migrate_database(settings.resolved_database_url)
    return Database(settings.resolved_database_url)


async def _seed_indexed_corpus(database: Database, *, count: int = 3) -> SeededCorpus:
    async with database.session_factory() as session:
        source = Source(
            type=SourceType.MANUAL,
            title="Source cerveau",
            raw_text="Texte source conserve.",
        )
        session.add(source)
        await session.flush()
        nodes = [
            KnowledgeNode(
                source_id=source.id,
                title=f"Connaissance {index}",
                content=f"Information autonome numero {index} sur la recuperation.",
            )
            for index in range(count)
        ]
        session.add_all(nodes)
        await session.flush()
        profile = EmbeddingProfile(
            provider="ollama",
            model_name="qwen3-embedding:0.6b",
            model_digest="embedding-digest-v1",
            dimensions=3,
            distance=EmbeddingDistance.COSINE,
            collection_name="brain-service-test-g1",
            semantic_text_version="title-content-v1",
            logical_generation=1,
            status=EmbeddingProfileStatus.ACTIVE,
            activated_at=utc_now(),
        )
        session.add(profile)
        await session.flush()
        active_nodes: list[ActiveEmbeddingNode] = []
        for index, node in enumerate(nodes):
            fingerprint = semantic_text_fingerprint(title=node.title, content=node.content)
            session.add(
                KnowledgeEmbedding(
                    knowledge_node_id=node.id,
                    embedding_profile_id=profile.id,
                    text_fingerprint=fingerprint,
                    status=KnowledgeEmbeddingStatus.INDEXED,
                    indexed_at=utc_now(),
                )
            )
            vector = [0.0, 0.0, 0.0]
            vector[index % 3] = 1.0
            active_nodes.append(
                ActiveEmbeddingNode(
                    id=node.id,
                    source_id=source.id,
                    title=node.title,
                    content=node.content,
                    tags=(),
                    text_fingerprint=fingerprint,
                    vector=tuple(vector),
                )
            )
        await session.commit()
        snapshot = ActiveEmbeddingProfile(
            id=profile.id,
            provider=profile.provider,
            model_name=profile.model_name,
            model_digest=profile.model_digest,
            dimensions=3,
            distance="cosine",
            collection_name=profile.collection_name,
            semantic_text_version=profile.semantic_text_version,
            logical_generation=profile.logical_generation,
        )
        return SeededCorpus(
            corpus=ActiveEmbeddingCorpus(profile=snapshot, nodes=tuple(active_nodes)),
            node_ids=tuple(node.id for node in nodes),
            source_id=source.id,
        )


def _fake_math(nodes: Sequence[object], _config: object) -> BrainMathResult:
    ordered = tuple(sorted(nodes, key=lambda node: str(node.id)))
    assert ordered
    # Les identifiants mathematiques sont reproductibles entre deux profils.
    root_id = UUID("00000000-0000-5000-8000-000000000001")
    theme_id = UUID("00000000-0000-5000-8000-000000000002")
    member_ids = tuple(node.id for node in ordered)
    centroid = tuple(
        sum(node.vector[index] for node in ordered) / len(ordered)
        for index in range(len(ordered[0].vector))
    )
    clusters = (
        MathCluster(
            id=root_id,
            parent_id=None,
            level=0,
            member_ids=member_ids,
            centroid=centroid,
            representative_ids=member_ids[:2],
            label="Second Brain",
            x=0.0,
            y=0.0,
        ),
        MathCluster(
            id=theme_id,
            parent_id=root_id,
            level=1,
            member_ids=member_ids,
            centroid=centroid,
            representative_ids=member_ids[:2],
            label="Recuperation",
            x=0.0,
            y=0.0,
        ),
    )
    layouts = tuple(
        MathNodeLayout(
            knowledge_node_id=node.id,
            cluster_id=theme_id,
            x=-0.5 + index * 0.4,
            y=0.25,
        )
        for index, node in enumerate(ordered)
    )
    edges = ()
    if len(ordered) > 1:
        source, target = sorted((ordered[0].id, ordered[1].id), key=str)
        edges = (
            MathEdge(
                source_node_id=source,
                target_node_id=target,
                cosine_score=0.75,
                tag_bonus=0.0,
                final_score=0.7125,
                mutual=True,
            ),
        )
    durations = BrainMathDurations(
        neighbors_seconds=0.01,
        pca_seconds=0.01,
        clustering_seconds=0.02,
        projection_seconds=0.03,
        labeling_seconds=0.001,
        total_seconds=0.071,
    )
    return BrainMathResult(
        nodes=layouts,
        edges=edges,
        clusters=clusters,
        stats=BrainMathStats(
            node_count=len(ordered),
            edge_count=len(edges),
            cluster_counts={0: 1, 1: 1},
            unassigned_count=0,
            cluster_size_min=len(ordered),
            cluster_size_mean=float(len(ordered)),
            cluster_size_max=len(ordered),
            similarity=SimilarityDistribution(
                minimum=0.75,
                mean=0.75,
                median=0.75,
                maximum=0.75,
            ),
            durations=durations,
            projection_algorithm="test_projection",
            pca_dimensions=min(3, len(ordered)),
        ),
    )


def _service(
    *,
    database: Database,
    settings: Settings,
    seeded: SeededCorpus,
    generator: FakeBrainGenerator,
) -> tuple[BrainService, FakeVectorService]:
    vectors = FakeVectorService(seeded.corpus)
    return (
        BrainService(
            database=database,
            vector_service=vectors,  # type: ignore[arg-type]
            generator=generator,
            settings=settings,
        ),
        vectors,
    )


async def _finish_job(
    runner: BrainRunner,
    service: BrainService,
    job_id: UUID,
) -> None:
    await service.run_job(job_id)
    await runner._mark_succeeded(job_id)


@pytest.mark.anyio
async def test_build_persists_snapshot_progress_and_llm_metrics(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        generator = FakeBrainGenerator()
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=generator,
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)

        job = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, job.id)

        async with database.session_factory() as session:
            stored_job = await session.get(ProcessingJob, job.id)
            profile = await session.get(BrainProfile, job.brain_profile_id)
            assert stored_job is not None
            assert profile is not None
            assert stored_job.status == ProcessingJobStatus.SUCCEEDED
            assert stored_job.progress_current == stored_job.progress_total == 5
            assert stored_job.progress_percent == 100
            assert stored_job.llm_call_count == 1
            assert stored_job.prompt_eval_count == 42
            assert stored_job.eval_count == 8
            assert profile.status == BrainProfileStatus.READY
            assert profile.knowledge_node_count == 3
            assert profile.cluster_count == 2
            assert profile.edge_count == 1
            assert profile.label_strategy == BrainLabelStrategy.OLLAMA
            assert profile.label_model_name == "qwen3.5:4b"
            assert (
                await session.scalar(
                    select(func.count(BrainCluster.id)).where(
                        BrainCluster.brain_profile_id == profile.id
                    )
                )
            ) == 2
            assert (
                await session.scalar(
                    select(func.count(BrainNodeLayout.knowledge_node_id)).where(
                        BrainNodeLayout.brain_profile_id == profile.id
                    )
                )
            ) == 3
            assert (
                await session.scalar(
                    select(func.count(BrainEdge.id)).where(BrainEdge.brain_profile_id == profile.id)
                )
            ) == 1
            generated_cluster = await session.scalar(
                select(BrainCluster).where(
                    BrainCluster.brain_profile_id == profile.id,
                    BrainCluster.level == 1,
                )
            )
            assert generated_cluster is not None
            assert generated_cluster.label == "Theme genere"
            assert generated_cluster.label_source == BrainLabelSource.OLLAMA
        assert generator.calls == 1
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_build_uses_deterministic_labels_when_ollama_is_unavailable(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        generator = FakeBrainGenerator(available=False)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=generator,
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)

        job = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, job.id)

        async with database.session_factory() as session:
            profile = await session.get(BrainProfile, job.brain_profile_id)
            assert profile is not None
            assert profile.status == BrainProfileStatus.READY
            assert profile.label_strategy == BrainLabelStrategy.DETERMINISTIC
            assert profile.label_model_name is None
            labels = list(
                (
                    await session.scalars(
                        select(BrainCluster.label_source).where(
                            BrainCluster.brain_profile_id == profile.id
                        )
                    )
                ).all()
            )
            assert labels == [BrainLabelSource.DETERMINISTIC] * 2
        assert generator.calls == 0
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_status_marks_ready_brain_stale_after_tag_change(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=FakeBrainGenerator(available=False),
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)
        job = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, job.id)

        async with database.session_factory() as session:
            tag = Tag(name="Fatigue", normalized_name="fatigue")
            session.add(tag)
            await session.flush()
            session.add(KnowledgeNodeTag(knowledge_node_id=seeded.node_ids[0], tag_id=tag.id))
            await session.commit()

        snapshot = await service.status()
        assert snapshot.state == "stale"
        assert any("tags ont change" in reason for reason in snapshot.stale_reasons)
        async with database.session_factory() as session:
            profile = await session.get(BrainProfile, job.brain_profile_id)
            assert profile is not None
            assert profile.status == BrainProfileStatus.STALE
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_status_detects_embedding_profile_change(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=FakeBrainGenerator(available=False),
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)
        job = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, job.id)

        async with database.session_factory() as session:
            current = await session.scalar(
                select(EmbeddingProfile).where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE
                )
            )
            assert current is not None
            current.status = EmbeddingProfileStatus.RETIRED
            await session.commit()
        async with database.session_factory() as session:
            replacement = EmbeddingProfile(
                provider="ollama",
                model_name="replacement-embedding",
                model_digest="embedding-digest-v2",
                dimensions=3,
                distance=EmbeddingDistance.COSINE,
                collection_name="brain-service-test-g2",
                semantic_text_version="title-content-v1",
                logical_generation=2,
                status=EmbeddingProfileStatus.ACTIVE,
                activated_at=utc_now(),
            )
            session.add(replacement)
            await session.flush()
            nodes = list((await session.scalars(select(KnowledgeNode))).all())
            for node in nodes:
                session.add(
                    KnowledgeEmbedding(
                        knowledge_node_id=node.id,
                        embedding_profile_id=replacement.id,
                        text_fingerprint=semantic_text_fingerprint(
                            title=node.title,
                            content=node.content,
                        ),
                        status=KnowledgeEmbeddingStatus.INDEXED,
                        indexed_at=utc_now(),
                    )
                )
            await session.commit()

        snapshot = await service.status()
        assert snapshot.state == "stale"
        assert "Le profil vectoriel actif a change." in snapshot.stale_reasons
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_failed_rebuild_keeps_previous_ready_profile(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=FakeBrainGenerator(available=False),
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)
        first = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, first.id)

        def fail_math(_nodes: Sequence[object], _config: object) -> BrainMathResult:
            raise RuntimeError("echec mathematique simule")

        monkeypatch.setattr("second_brain.services.brain.build_brain_math", fail_math)
        second = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        with pytest.raises(RuntimeError, match="echec mathematique"):
            await service.run_job(second.id)
        await runner._mark_failed(
            second.id,
            code="internal_error",
            error_type="RuntimeError",
            message="Une erreur interne a interrompu la construction du cerveau.",
        )

        async with database.session_factory() as session:
            previous = await session.get(BrainProfile, first.brain_profile_id)
            failed = await session.get(BrainProfile, second.brain_profile_id)
            assert previous is not None and previous.status == BrainProfileStatus.READY
            assert failed is not None and failed.status == BrainProfileStatus.ERROR
            assert failed.error_message is not None
            assert (
                await session.scalar(
                    select(func.count(BrainCluster.id)).where(
                        BrainCluster.brain_profile_id == previous.id
                    )
                )
            ) == 2
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_enqueue_is_idempotent_while_build_is_pending(
    settings: Settings,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        service, vectors = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=FakeBrainGenerator(available=False),
        )
        runner = BrainRunner(database=database, service=service, settings=settings)

        first = await runner.enqueue(ProcessingJobKind.BUILD_BRAIN)
        second = await runner.enqueue(ProcessingJobKind.BUILD_BRAIN)

        assert second.id == first.id
        assert second.brain_profile_id == first.brain_profile_id
        # Le second clic reutilise le job actif sans relire les vecteurs Qdrant.
        assert vectors.calls == 1
        async with database.session_factory() as session:
            assert (
                await session.scalar(
                    select(func.count(ProcessingJob.id)).where(
                        ProcessingJob.kind == ProcessingJobKind.BUILD_BRAIN
                    )
                )
            ) == 1
            assert (await session.scalar(select(func.count(BrainProfile.id)))) == 1
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_running_job_is_recovered_and_can_complete_after_restart(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=FakeBrainGenerator(available=False),
        )
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)
        job = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        async with database.session_factory() as session:
            stored = await session.get(ProcessingJob, job.id)
            assert stored is not None
            stored.status = ProcessingJobStatus.RUNNING
            stored.stage = "mathematical_model"
            stored.error_code = "interrupted"
            stored.error_message = "Ancienne erreur"
            await session.commit()

        restarted_runner = BrainRunner(database=database, service=service, settings=settings)
        await restarted_runner._recover_interrupted_jobs()
        async with database.session_factory() as session:
            recovered = await session.get(ProcessingJob, job.id)
            assert recovered is not None
            assert recovered.status == ProcessingJobStatus.PENDING
            assert recovered.stage == "queued"
            assert recovered.error_code is None
            assert recovered.error_message is None

        claimed = await restarted_runner._claim_next_job()
        assert claimed == job.id
        await service.run_job(job.id)
        await restarted_runner._mark_succeeded(job.id)
        async with database.session_factory() as session:
            completed = await session.get(ProcessingJob, job.id)
            profile = await session.get(BrainProfile, job.brain_profile_id)
            assert completed is not None
            assert completed.status == ProcessingJobStatus.SUCCEEDED
            assert completed.attempt_count == 1
            assert profile is not None and profile.status == BrainProfileStatus.READY
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_successive_rebuilds_keep_deterministic_cluster_ids_per_profile(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=FakeBrainGenerator(available=False),
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)

        first = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, first.id)
        second = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, second.id)

        assert first.brain_profile_id != second.brain_profile_id
        async with database.session_factory() as session:
            old_profile = await session.get(BrainProfile, first.brain_profile_id)
            active_profile = await session.get(BrainProfile, second.brain_profile_id)
            assert old_profile is not None
            assert active_profile is not None
            assert old_profile.status == BrainProfileStatus.STALE
            assert active_profile.status == BrainProfileStatus.READY
            first_cluster_ids = set(
                (
                    await session.scalars(
                        select(BrainCluster.id).where(
                            BrainCluster.brain_profile_id == old_profile.id
                        )
                    )
                ).all()
            )
            second_cluster_ids = set(
                (
                    await session.scalars(
                        select(BrainCluster.id).where(
                            BrainCluster.brain_profile_id == active_profile.id
                        )
                    )
                ).all()
            )
            assert (
                first_cluster_ids
                == second_cluster_ids
                == {
                    UUID("00000000-0000-5000-8000-000000000001"),
                    UUID("00000000-0000-5000-8000-000000000002"),
                }
            )
            assert (await session.scalar(select(func.count(BrainCluster.id)))) == 4
    finally:
        await database.dispose()


@pytest.mark.anyio
async def test_relabel_updates_non_root_label_and_job_metrics_without_graph_changes(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = await _database(settings)
    try:
        seeded = await _seed_indexed_corpus(database)
        generator = FakeBrainGenerator(available=False)
        service, _ = _service(
            database=database,
            settings=settings,
            seeded=seeded,
            generator=generator,
        )
        runner = BrainRunner(database=database, service=service, settings=settings)
        monkeypatch.setattr("second_brain.services.brain.build_brain_math", _fake_math)
        build = await service.prepare_job(ProcessingJobKind.BUILD_BRAIN)
        await _finish_job(runner, service, build.id)

        async with database.session_factory() as session:
            root_before = await session.scalar(
                select(BrainCluster).where(
                    BrainCluster.brain_profile_id == build.brain_profile_id,
                    BrainCluster.level == 0,
                )
            )
            theme_before = await session.scalar(
                select(BrainCluster).where(
                    BrainCluster.brain_profile_id == build.brain_profile_id,
                    BrainCluster.level == 1,
                )
            )
            assert root_before is not None and theme_before is not None
            root_label_before = root_before.label
            theme_label_before = theme_before.label
            layout_snapshot = tuple(
                (
                    row.knowledge_node_id,
                    row.cluster_id,
                    row.x,
                    row.y,
                    row.is_unassigned,
                )
                for row in (
                    await session.scalars(
                        select(BrainNodeLayout)
                        .where(BrainNodeLayout.brain_profile_id == build.brain_profile_id)
                        .order_by(BrainNodeLayout.knowledge_node_id)
                    )
                ).all()
            )
            edge_snapshot = tuple(
                (
                    row.source_node_id,
                    row.target_node_id,
                    row.cosine_score,
                    row.final_score,
                )
                for row in (
                    await session.scalars(
                        select(BrainEdge).where(
                            BrainEdge.brain_profile_id == build.brain_profile_id
                        )
                    )
                ).all()
            )

        generator.available = True
        relabel = await service.prepare_job(ProcessingJobKind.RELABEL_BRAIN)
        await _finish_job(runner, service, relabel.id)

        async with database.session_factory() as session:
            stored_job = await session.get(ProcessingJob, relabel.id)
            profile = await session.get(BrainProfile, build.brain_profile_id)
            root_after = await session.scalar(
                select(BrainCluster).where(
                    BrainCluster.brain_profile_id == build.brain_profile_id,
                    BrainCluster.level == 0,
                )
            )
            theme_after = await session.scalar(
                select(BrainCluster).where(
                    BrainCluster.brain_profile_id == build.brain_profile_id,
                    BrainCluster.level == 1,
                )
            )
            assert stored_job is not None
            assert profile is not None
            assert root_after is not None and theme_after is not None
            assert stored_job.status == ProcessingJobStatus.SUCCEEDED
            assert stored_job.progress_percent == 100
            assert stored_job.llm_call_count == 1
            assert stored_job.prompt_eval_count == 42
            assert stored_job.eval_count == 8
            assert profile.label_strategy == BrainLabelStrategy.OLLAMA
            assert profile.label_model_name == "qwen3.5:4b"
            assert root_after.label == root_label_before
            assert root_after.label_source == BrainLabelSource.DETERMINISTIC
            assert theme_after.label != theme_label_before
            assert theme_after.label == "Theme genere"
            assert theme_after.label_source == BrainLabelSource.OLLAMA
            layouts_after = tuple(
                (
                    row.knowledge_node_id,
                    row.cluster_id,
                    row.x,
                    row.y,
                    row.is_unassigned,
                )
                for row in (
                    await session.scalars(
                        select(BrainNodeLayout)
                        .where(BrainNodeLayout.brain_profile_id == build.brain_profile_id)
                        .order_by(BrainNodeLayout.knowledge_node_id)
                    )
                ).all()
            )
            edges_after = tuple(
                (
                    row.source_node_id,
                    row.target_node_id,
                    row.cosine_score,
                    row.final_score,
                )
                for row in (
                    await session.scalars(
                        select(BrainEdge).where(
                            BrainEdge.brain_profile_id == build.brain_profile_id
                        )
                    )
                ).all()
            )
            assert layouts_after == layout_snapshot
            assert edges_after == edge_snapshot
        assert generator.calls == 1
    finally:
        await database.dispose()
