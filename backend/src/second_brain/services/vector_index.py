from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from second_brain.core.config import Settings
from second_brain.db.base import utc_now
from second_brain.db.models.embedding import (
    EmbeddingDistance,
    EmbeddingProfile,
    EmbeddingProfileStatus,
    KnowledgeEmbedding,
    KnowledgeEmbeddingStatus,
)
from second_brain.db.models.knowledge import KnowledgeEvidence, KnowledgeNode
from second_brain.db.models.processing import ProcessingJob, ProcessingJobKind
from second_brain.db.models.taxonomy import KnowledgeNodeTag
from second_brain.db.session import Database
from second_brain.llm.errors import OllamaError
from second_brain.llm.schemas import OllamaReadiness
from second_brain.vector.embeddings import (
    EmbeddingCallContext,
    EmbeddingCallMetrics,
    EmbeddingProvider,
)
from second_brain.vector.semantic_text import (
    SEMANTIC_TEXT_VERSION,
    build_semantic_text,
    semantic_text_fingerprint,
)
from second_brain.vector.store import (
    VectorCollectionInfo,
    VectorPoint,
    VectorSearchHit,
    VectorStore,
    VectorStoreCompatibilityError,
    VectorStoreCorruptedError,
    VectorStoreError,
    VectorStoreUnavailableError,
)

logger = logging.getLogger(__name__)

VectorIndexState = Literal[
    "empty",
    "not_built",
    "building",
    "ready",
    "stale",
    "incompatible",
    "unavailable",
    "corrupt",
]

VECTOR_JOB_KINDS = (
    ProcessingJobKind.INDEX_KNOWLEDGE,
    ProcessingJobKind.REBUILD_VECTOR_INDEX,
)


class VectorIndexError(RuntimeError):
    code = "vector_index_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class VectorIndexNotBuiltError(VectorIndexError):
    code = "vector_index_not_built"


class VectorIndexIncompatibleError(VectorIndexError):
    code = "vector_index_incompatible"


class VectorIndexBusyError(VectorIndexError):
    code = "vector_index_busy"


class VectorIndexUnavailableError(VectorIndexError):
    code = "vector_index_unavailable"


@dataclass(frozen=True, slots=True)
class VectorIndexSnapshot:
    state: VectorIndexState
    readiness: OllamaReadiness
    total_nodes: int
    indexed_nodes: int
    pending_or_stale_nodes: int
    failed_nodes: int
    profile: EmbeddingProfile | None
    active_job: ProcessingJob | None
    orphan_points: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    score: float
    node: KnowledgeNode


@dataclass(frozen=True, slots=True)
class SemanticSearchResults:
    query: str
    profile: EmbeddingProfile | None
    items: tuple[SemanticSearchResult, ...]


@dataclass(frozen=True, slots=True)
class _NodeInput:
    id: UUID
    source_id: UUID
    title: str
    content: str

    @property
    def fingerprint(self) -> str:
        return semantic_text_fingerprint(title=self.title, content=self.content)

    @property
    def semantic_text(self) -> str:
        return build_semantic_text(title=self.title, content=self.content)


class VectorIndexService:
    """Synchronize the reconstructible Qdrant index with SQLite knowledge."""

    def __init__(
        self,
        *,
        database: Database,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings,
    ) -> None:
        self._database = database
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._settings = settings
        self._batch_size = settings.embedding_batch_size
        self._enqueue_lock = asyncio.Lock()

    @property
    def configured_model(self) -> str:
        return self._embedding_provider.configured_model

    async def get_embedding_readiness(self) -> OllamaReadiness:
        return await self._embedding_provider.get_readiness()

    async def status(self) -> VectorIndexSnapshot:
        readiness = await self._embedding_provider.get_readiness()
        async with self._database.session_factory() as session:
            total = await self._count_nodes(session)
            active_profile = await self._active_profile(session)
            building_profile = await self._building_profile(session)
            active_job = await self._active_vector_job(session)
            profile = active_profile or building_profile
            indexed, pending, failed = await self._profile_counts(
                session,
                profile=profile,
                total_nodes=total,
            )

        if total == 0 and profile is None:
            return VectorIndexSnapshot(
                state="empty",
                readiness=readiness,
                total_nodes=0,
                indexed_nodes=0,
                pending_or_stale_nodes=0,
                failed_nodes=0,
                profile=None,
                active_job=active_job,
            )
        if active_profile is None:
            return VectorIndexSnapshot(
                state="building" if building_profile or active_job else "not_built",
                readiness=readiness,
                total_nodes=total,
                indexed_nodes=indexed,
                pending_or_stale_nodes=pending,
                failed_nodes=failed,
                profile=building_profile,
                active_job=active_job,
            )

        if not _model_names_match(active_profile.model_name, self.configured_model):
            return VectorIndexSnapshot(
                state="building" if active_job else "incompatible",
                readiness=readiness,
                total_nodes=total,
                indexed_nodes=indexed,
                pending_or_stale_nodes=pending,
                failed_nodes=failed,
                profile=active_profile,
                active_job=active_job,
                error=(
                    "Le modele d'embedding configure differe de celui de l'index actif. "
                    "Reconstruisez l'index avant de poursuivre."
                ),
            )

        store_state, store_error = await self._inspect_active_collection(active_profile)
        if store_state is not None:
            return VectorIndexSnapshot(
                state=store_state,
                readiness=readiness,
                total_nodes=total,
                indexed_nodes=indexed,
                pending_or_stale_nodes=pending,
                failed_nodes=failed,
                profile=active_profile,
                active_job=active_job,
                error=store_error,
            )
        try:
            indexed, pending, failed, orphan_points = await self._point_aware_profile_counts(
                active_profile
            )
        except VectorStoreCompatibilityError as error:
            store_state, store_error = "corrupt", error.message
        except VectorStoreCorruptedError as error:
            store_state, store_error = "corrupt", error.message
        except (VectorStoreUnavailableError, VectorStoreError) as error:
            store_state, store_error = "unavailable", error.message
        else:
            store_state = None
            store_error = None
        if store_state is not None:
            return VectorIndexSnapshot(
                state=store_state,
                readiness=readiness,
                total_nodes=total,
                indexed_nodes=indexed,
                pending_or_stale_nodes=pending,
                failed_nodes=failed,
                profile=active_profile,
                active_job=active_job,
                orphan_points=0,
                error=store_error,
            )

        state: VectorIndexState
        if active_job is not None:
            state = "building"
        elif pending or failed or orphan_points:
            state = "stale"
        else:
            state = "ready"
        return VectorIndexSnapshot(
            state=state,
            readiness=readiness,
            total_nodes=total,
            indexed_nodes=indexed,
            pending_or_stale_nodes=pending,
            failed_nodes=failed,
            profile=active_profile,
            active_job=active_job,
            orphan_points=orphan_points,
        )

    async def prepare_job(self, kind: ProcessingJobKind) -> ProcessingJob:
        async with self._enqueue_lock:
            return await self._prepare_job_locked(kind)

    async def _prepare_job_locked(self, kind: ProcessingJobKind) -> ProcessingJob:
        if kind not in VECTOR_JOB_KINDS:
            raise ValueError("Le type de traitement vectoriel est invalide.")

        async with self._database.session_factory() as session:
            existing = await self._active_vector_job(session)
            if existing is not None:
                if existing.kind == kind:
                    return existing
                raise VectorIndexBusyError("Une autre operation d'indexation est deja en cours.")

            active_profile = await self._active_profile(session)
            if (
                kind == ProcessingJobKind.INDEX_KNOWLEDGE
                and active_profile is not None
                and not _model_names_match(active_profile.model_name, self.configured_model)
            ):
                raise VectorIndexIncompatibleError(
                    "Le modele d'embedding a change. Utilisez Reconstruire l'index."
                )

            profile: EmbeddingProfile | None = None
            if kind == ProcessingJobKind.INDEX_KNOWLEDGE:
                profile = active_profile or await self._building_profile(
                    session,
                    configured_only=True,
                )
            else:
                profile = await self._building_profile(session, configured_only=True)

            total = await self._count_nodes(session)
            job = ProcessingJob(
                source_id=None,
                embedding_profile_id=profile.id if profile else None,
                kind=kind,
                stage="queued",
                progress_current=0,
                progress_total=total,
                progress_percent=0,
                progress_message="Indexation en attente.",
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

    async def run_job(self, job_id: UUID) -> None:
        job, profile, nodes = await self._prepare_run(job_id)
        if not nodes:
            if profile is not None and profile.dimensions is not None:
                await self._remove_orphan_points(profile)
            await self._update_progress(job_id, current=0, total=0, message="Aucune connaissance.")
            return

        profile = profile or await self._create_building_profile(job_id)
        await self._update_progress(
            job_id,
            current=0,
            total=len(nodes),
            message=f"Indexation : 0 / {len(nodes)} connaissances.",
        )

        completed = 0
        for batch_index, offset in enumerate(range(0, len(nodes), self._batch_size), start=1):
            batch = nodes[offset : offset + self._batch_size]
            failed_nodes = batch
            try:
                recovered, pending = await self._partition_batch(profile, batch)
                failed_nodes = pending
                if recovered:
                    await self._checkpoint_recovered(profile, recovered)
                if pending:
                    await self._embed_and_checkpoint(
                        job=job,
                        profile=profile,
                        nodes=pending,
                        batch_index=batch_index,
                        batch_total=(len(nodes) + self._batch_size - 1) // self._batch_size,
                    )
            except Exception as error:
                if failed_nodes:
                    await self._mark_batch_failed(profile, failed_nodes, error)
                profile.error_message = _safe_error_message(error)
                async with self._database.session_factory() as session:
                    persisted_profile = await session.get(EmbeddingProfile, profile.id)
                    if persisted_profile is not None:
                        persisted_profile.error_message = profile.error_message
                        await session.commit()
                raise

            completed += len(batch)
            await self._update_progress(
                job_id,
                current=completed,
                total=len(nodes),
                message=f"Indexation : {completed} / {len(nodes)} connaissances.",
            )

        await self._remove_orphan_points(profile)
        if profile.status == EmbeddingProfileStatus.BUILDING:
            await self._activate_profile(profile.id)

    async def search(self, query: str, *, top_k: int) -> SemanticSearchResults:
        async with self._database.session_factory() as session:
            profile = await self._active_profile(session)
            total = await self._count_nodes(session)
        if profile is None:
            if total == 0:
                return SemanticSearchResults(query=query, profile=None, items=())
            raise VectorIndexNotBuiltError(
                "L'index semantique n'existe pas encore. Indexez les connaissances."
            )
        self._validate_profile_model(profile)
        if profile.dimensions is None:
            raise VectorIndexIncompatibleError(
                "La dimension de l'index est inconnue. Reconstruisez l'index."
            )

        embedded = await self._embedding_provider.embed(
            [query],
            context=EmbeddingCallContext(operation="semantic_search"),
        )
        if embedded.dimension != profile.dimensions:
            raise VectorIndexIncompatibleError(
                "Le modele produit une dimension differente de l'index actif. "
                "Reconstruisez l'index."
            )
        hits = await self._vector_store.search(
            profile.collection_name,
            embedded.vectors[0],
            limit=min(50, max(top_k, top_k * 3)),
        )
        items = await self._load_search_results(profile, hits, top_k=top_k)
        return SemanticSearchResults(query=query, profile=profile, items=tuple(items))

    async def _prepare_run(
        self,
        job_id: UUID,
    ) -> tuple[ProcessingJob, EmbeddingProfile | None, list[_NodeInput]]:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None or job.kind not in VECTOR_JOB_KINDS:
                raise VectorIndexError("Le traitement d'indexation est introuvable.")
            profile = (
                await session.get(EmbeddingProfile, job.embedding_profile_id)
                if job.embedding_profile_id
                else None
            )
            if profile is None and job.kind == ProcessingJobKind.INDEX_KNOWLEDGE:
                profile = await self._active_profile(session)
            if profile is not None:
                self._validate_profile_model(profile)

            rows = (
                await session.execute(
                    select(
                        KnowledgeNode.id,
                        KnowledgeNode.source_id,
                        KnowledgeNode.title,
                        KnowledgeNode.content,
                    )
                    .where(KnowledgeNode.created_at <= job.created_at)
                    .order_by(KnowledgeNode.id.asc())
                )
            ).all()
            nodes = [
                _NodeInput(id=row.id, source_id=row.source_id, title=row.title, content=row.content)
                for row in rows
            ]
            return job, profile, nodes

    async def _create_building_profile(self, job_id: UUID) -> EmbeddingProfile:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                raise VectorIndexError("Le traitement d'indexation est introuvable.")
            latest_generation = await session.scalar(
                select(func.max(EmbeddingProfile.logical_generation))
            )
            profile_id = uuid4()
            generation = int(latest_generation or 0) + 1
            profile = EmbeddingProfile(
                id=profile_id,
                provider="ollama",
                model_name=self.configured_model,
                model_digest=None,
                dimensions=None,
                distance=EmbeddingDistance.COSINE,
                collection_name=f"second_brain_nodes_g{generation}_{profile_id.hex[:8]}",
                semantic_text_version=SEMANTIC_TEXT_VERSION,
                logical_generation=generation,
                status=EmbeddingProfileStatus.BUILDING,
            )
            session.add(profile)
            job.embedding_profile_id = profile.id
            await session.commit()
            await session.refresh(profile)
            return profile

    async def _partition_batch(
        self,
        profile: EmbeddingProfile,
        nodes: list[_NodeInput],
    ) -> tuple[list[_NodeInput], list[_NodeInput]]:
        if profile.dimensions is None:
            return [], nodes
        info = await self._vector_store.inspect_collection(profile.collection_name)
        if info is None:
            records = await self._embedding_records(profile.id, [node.id for node in nodes])
            if any(
                record.status == KnowledgeEmbeddingStatus.INDEXED for record in records.values()
            ):
                raise VectorStoreCorruptedError(
                    "La collection Qdrant attendue est absente. Reconstruisez l'index."
                )
            return [], nodes
        self._validate_collection(profile, info)

        stored = await self._vector_store.retrieve(
            profile.collection_name,
            [node.id for node in nodes],
        )
        stored_by_id = {point.knowledge_node_id: point for point in stored}
        recovered: list[_NodeInput] = []
        pending: list[_NodeInput] = []
        for node in nodes:
            point = stored_by_id.get(node.id)
            if point is not None and point.fingerprint == node.fingerprint:
                recovered.append(node)
            else:
                pending.append(node)
        return recovered, pending

    async def _embed_and_checkpoint(
        self,
        *,
        job: ProcessingJob,
        profile: EmbeddingProfile,
        nodes: list[_NodeInput],
        batch_index: int,
        batch_total: int,
    ) -> None:
        captured_metrics = []
        try:
            result = await self._embedding_provider.embed(
                [node.semantic_text for node in nodes],
                context=EmbeddingCallContext(
                    processing_job_id=job.id,
                    operation="index_knowledge",
                    batch_index=batch_index,
                    batch_total=batch_total,
                ),
                metrics_callback=captured_metrics.append,
            )
        finally:
            if captured_metrics:
                await self._record_embedding_metrics(job.id, captured_metrics)
        if not _model_names_match(result.model, profile.model_name):
            raise VectorIndexIncompatibleError(
                "Ollama a utilise un modele different du profil d'indexation."
            )
        if profile.dimensions is None:
            await self._vector_store.ensure_collection(profile.collection_name, result.dimension)
            async with self._database.session_factory() as session:
                persisted = await session.get(EmbeddingProfile, profile.id)
                if persisted is None:
                    raise VectorIndexError("Le profil d'indexation a disparu.")
                persisted.dimensions = result.dimension
                persisted.error_message = None
                await session.commit()
            profile.dimensions = result.dimension
        elif result.dimension != profile.dimensions:
            raise VectorIndexIncompatibleError(
                "La dimension produite par Ollama ne correspond pas au profil. "
                "Reconstruisez l'index."
            )
        else:
            await self._vector_store.ensure_collection(
                profile.collection_name,
                profile.dimensions,
            )

        points = [
            VectorPoint(
                knowledge_node_id=node.id,
                source_id=node.source_id,
                fingerprint=node.fingerprint,
                vector=vector,
            )
            for node, vector in zip(nodes, result.vectors, strict=True)
        ]
        await self._vector_store.upsert(profile.collection_name, points)

        async with self._database.session_factory() as session:
            now = utc_now()
            for node in nodes:
                record = await session.get(KnowledgeEmbedding, (node.id, profile.id))
                if record is None:
                    record = KnowledgeEmbedding(
                        knowledge_node_id=node.id,
                        embedding_profile_id=profile.id,
                        text_fingerprint=node.fingerprint,
                        attempt_count=1,
                    )
                    session.add(record)
                else:
                    record.attempt_count += 1
                record.text_fingerprint = node.fingerprint
                record.status = KnowledgeEmbeddingStatus.INDEXED
                record.error_message = None
                record.indexed_at = now

            persisted_job = await session.get(ProcessingJob, job.id)
            if persisted_job is not None:
                persisted_job.last_activity_at = now
            await session.commit()

    async def _record_embedding_metrics(
        self,
        job_id: UUID,
        metrics: list[EmbeddingCallMetrics],
    ) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            for metric in metrics:
                job.embedding_batch_count += 1
                job.embedding_item_count += max(0, int(metric.batch_size))
                job.embedding_duration_ms += max(
                    0,
                    int(metric.duration_seconds * 1000),
                )
                job.embedding_total_duration_ns += max(
                    0,
                    int(metric.total_duration_ns or 0),
                )
                job.embedding_prompt_eval_count += max(
                    0,
                    int(metric.prompt_eval_count or 0),
                )
            job.last_activity_at = utc_now()
            await session.commit()

    async def _checkpoint_recovered(
        self,
        profile: EmbeddingProfile,
        nodes: list[_NodeInput],
    ) -> None:
        async with self._database.session_factory() as session:
            now = utc_now()
            for node in nodes:
                record = await session.get(KnowledgeEmbedding, (node.id, profile.id))
                if record is None:
                    record = KnowledgeEmbedding(
                        knowledge_node_id=node.id,
                        embedding_profile_id=profile.id,
                        text_fingerprint=node.fingerprint,
                    )
                    session.add(record)
                record.text_fingerprint = node.fingerprint
                record.status = KnowledgeEmbeddingStatus.INDEXED
                record.error_message = None
                record.indexed_at = record.indexed_at or now
            await session.commit()

    async def _mark_batch_failed(
        self,
        profile: EmbeddingProfile,
        nodes: list[_NodeInput],
        error: Exception,
    ) -> None:
        message = _safe_error_message(error)
        async with self._database.session_factory() as session:
            for node in nodes:
                record = await session.get(KnowledgeEmbedding, (node.id, profile.id))
                if record is None:
                    record = KnowledgeEmbedding(
                        knowledge_node_id=node.id,
                        embedding_profile_id=profile.id,
                        text_fingerprint=node.fingerprint,
                        attempt_count=1,
                    )
                    session.add(record)
                else:
                    record.attempt_count += 1
                record.text_fingerprint = node.fingerprint
                record.status = KnowledgeEmbeddingStatus.FAILED
                record.error_message = message
            await session.commit()

    async def _remove_orphan_points(self, profile: EmbeddingProfile) -> None:
        if profile.dimensions is None:
            return
        point_ids = set(await self._vector_store.list_point_ids(profile.collection_name))
        if not point_ids:
            return
        async with self._database.session_factory() as session:
            node_ids = set((await session.scalars(select(KnowledgeNode.id))).all())
        orphan_ids = sorted(point_ids - node_ids, key=str)
        if orphan_ids:
            await self._vector_store.delete(profile.collection_name, orphan_ids)

    async def _activate_profile(self, profile_id: UUID) -> None:
        async with self._database.session_factory() as session:
            profile = await session.get(EmbeddingProfile, profile_id)
            if profile is None or profile.dimensions is None:
                raise VectorIndexError("Le profil vectoriel ne peut pas etre active.")
            await session.execute(
                update(EmbeddingProfile)
                .where(
                    EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE,
                    EmbeddingProfile.id != profile.id,
                )
                .values(status=EmbeddingProfileStatus.RETIRED)
            )
            profile.status = EmbeddingProfileStatus.ACTIVE
            profile.activated_at = utc_now()
            profile.error_message = None
            await session.commit()

    async def _load_search_results(
        self,
        profile: EmbeddingProfile,
        hits: list[VectorSearchHit],
        *,
        top_k: int,
    ) -> list[SemanticSearchResult]:
        if not hits:
            return []
        ids = [hit.knowledge_node_id for hit in hits]
        async with self._database.session_factory() as session:
            nodes = list(
                (
                    await session.scalars(
                        select(KnowledgeNode)
                        .where(KnowledgeNode.id.in_(ids))
                        .options(
                            selectinload(KnowledgeNode.source),
                            selectinload(KnowledgeNode.tag_links).selectinload(
                                KnowledgeNodeTag.tag
                            ),
                            selectinload(KnowledgeNode.evidence).selectinload(
                                KnowledgeEvidence.passage
                            ),
                        )
                    )
                )
                .unique()
                .all()
            )
            records = {
                record.knowledge_node_id: record
                for record in (
                    await session.scalars(
                        select(KnowledgeEmbedding).where(
                            KnowledgeEmbedding.embedding_profile_id == profile.id,
                            KnowledgeEmbedding.knowledge_node_id.in_(ids),
                        )
                    )
                ).all()
            }

        nodes_by_id = {node.id: node for node in nodes}
        results: list[SemanticSearchResult] = []
        for hit in hits:
            node = nodes_by_id.get(hit.knowledge_node_id)
            record = records.get(hit.knowledge_node_id)
            if node is None or record is None:
                continue
            current_fingerprint = semantic_text_fingerprint(
                title=node.title,
                content=node.content,
            )
            if (
                record.status != KnowledgeEmbeddingStatus.INDEXED
                or record.text_fingerprint != current_fingerprint
                or hit.fingerprint != current_fingerprint
            ):
                continue
            results.append(SemanticSearchResult(score=hit.score, node=node))
            if len(results) >= top_k:
                break
        return results

    async def _profile_counts(
        self,
        session: AsyncSession,
        *,
        profile: EmbeddingProfile | None,
        total_nodes: int,
    ) -> tuple[int, int, int]:
        if profile is None:
            return 0, total_nodes, 0
        nodes = (
            await session.execute(
                select(KnowledgeNode.id, KnowledgeNode.title, KnowledgeNode.content)
            )
        ).all()
        records = {
            record.knowledge_node_id: record
            for record in (
                await session.scalars(
                    select(KnowledgeEmbedding).where(
                        KnowledgeEmbedding.embedding_profile_id == profile.id
                    )
                )
            ).all()
        }
        indexed = 0
        failed = 0
        for node in nodes:
            fingerprint = semantic_text_fingerprint(title=node.title, content=node.content)
            record = records.get(node.id)
            if (
                record is not None
                and record.status == KnowledgeEmbeddingStatus.INDEXED
                and record.text_fingerprint == fingerprint
            ):
                indexed += 1
            elif (
                record is not None
                and record.status == KnowledgeEmbeddingStatus.FAILED
                and record.text_fingerprint == fingerprint
            ):
                failed += 1
        return indexed, max(0, total_nodes - indexed - failed), failed

    async def _point_aware_profile_counts(
        self,
        profile: EmbeddingProfile,
    ) -> tuple[int, int, int, int]:
        async with self._database.session_factory() as session:
            nodes = (
                await session.execute(
                    select(KnowledgeNode.id, KnowledgeNode.title, KnowledgeNode.content)
                )
            ).all()
            records = {
                record.knowledge_node_id: record
                for record in (
                    await session.scalars(
                        select(KnowledgeEmbedding).where(
                            KnowledgeEmbedding.embedding_profile_id == profile.id
                        )
                    )
                ).all()
            }
        stored = await self._vector_store.retrieve(
            profile.collection_name,
            [node.id for node in nodes],
        )
        points = {point.knowledge_node_id: point for point in stored}
        qdrant_ids = await self._vector_store.list_point_ids(profile.collection_name)
        sqlite_ids = {node.id for node in nodes}
        orphan_points = len(qdrant_ids - sqlite_ids)
        indexed = 0
        failed = 0
        for node in nodes:
            fingerprint = semantic_text_fingerprint(title=node.title, content=node.content)
            record = records.get(node.id)
            point = points.get(node.id)
            if (
                record is not None
                and record.status == KnowledgeEmbeddingStatus.INDEXED
                and record.text_fingerprint == fingerprint
                and point is not None
                and point.fingerprint == fingerprint
            ):
                indexed += 1
            elif (
                record is not None
                and record.status == KnowledgeEmbeddingStatus.FAILED
                and record.text_fingerprint == fingerprint
            ):
                failed += 1
        return indexed, max(0, len(nodes) - indexed - failed), failed, orphan_points

    async def _embedding_records(
        self,
        profile_id: UUID,
        node_ids: list[UUID],
    ) -> dict[UUID, KnowledgeEmbedding]:
        async with self._database.session_factory() as session:
            return {
                record.knowledge_node_id: record
                for record in (
                    await session.scalars(
                        select(KnowledgeEmbedding).where(
                            KnowledgeEmbedding.embedding_profile_id == profile_id,
                            KnowledgeEmbedding.knowledge_node_id.in_(node_ids),
                        )
                    )
                ).all()
            }

    async def _inspect_active_collection(
        self,
        profile: EmbeddingProfile,
    ) -> tuple[VectorIndexState | None, str | None]:
        if profile.dimensions is None:
            return "corrupt", "La dimension de l'index actif est absente."
        try:
            info = await self._vector_store.inspect_collection(profile.collection_name)
            if info is None:
                return "corrupt", "La collection Qdrant active est absente."
            self._validate_collection(profile, info)
        except VectorStoreCompatibilityError as error:
            return "corrupt", error.message
        except VectorStoreCorruptedError as error:
            return "corrupt", error.message
        except (VectorStoreUnavailableError, VectorStoreError) as error:
            return "unavailable", error.message
        return None, None

    @staticmethod
    def _validate_collection(
        profile: EmbeddingProfile,
        info: VectorCollectionInfo,
    ) -> None:
        if profile.dimensions is None:
            raise VectorIndexIncompatibleError("La dimension du profil est inconnue.")
        if info.dimension != profile.dimensions or info.distance != "cosine":
            raise VectorStoreCompatibilityError(
                "La collection Qdrant a une dimension ou une metrique incompatible."
            )

    def _validate_profile_model(self, profile: EmbeddingProfile) -> None:
        if not _model_names_match(profile.model_name, self.configured_model):
            raise VectorIndexIncompatibleError(
                "Le modele d'embedding configure differe de l'index. Reconstruisez l'index."
            )
        if profile.semantic_text_version != SEMANTIC_TEXT_VERSION:
            raise VectorIndexIncompatibleError(
                "La construction du texte semantique a change. Reconstruisez l'index."
            )

    async def _update_progress(
        self,
        job_id: UUID,
        *,
        current: int,
        total: int,
        message: str,
    ) -> None:
        async with self._database.session_factory() as session:
            job = await session.get(ProcessingJob, job_id)
            if job is None:
                return
            job.stage = "indexing"
            job.progress_current = current
            job.progress_total = total
            job.progress_percent = 100 if total == 0 else min(100, current * 100 // total)
            job.progress_message = message
            job.last_activity_at = utc_now()
            await session.commit()

    @staticmethod
    async def _count_nodes(session: AsyncSession) -> int:
        return int((await session.scalar(select(func.count(KnowledgeNode.id)))) or 0)

    @staticmethod
    async def _active_profile(session: AsyncSession) -> EmbeddingProfile | None:
        return await session.scalar(
            select(EmbeddingProfile)
            .where(EmbeddingProfile.status == EmbeddingProfileStatus.ACTIVE)
            .order_by(EmbeddingProfile.logical_generation.desc())
            .limit(1)
        )

    async def _building_profile(
        self,
        session: AsyncSession,
        *,
        configured_only: bool = False,
    ) -> EmbeddingProfile | None:
        statement = select(EmbeddingProfile).where(
            EmbeddingProfile.status == EmbeddingProfileStatus.BUILDING
        )
        if configured_only:
            statement = statement.where(
                EmbeddingProfile.model_name == self.configured_model,
                EmbeddingProfile.semantic_text_version == SEMANTIC_TEXT_VERSION,
            )
        return await session.scalar(
            statement.order_by(EmbeddingProfile.logical_generation.desc()).limit(1)
        )

    @staticmethod
    async def _active_vector_job(session: AsyncSession) -> ProcessingJob | None:
        from second_brain.db.models.processing import ProcessingJobStatus

        return await session.scalar(
            select(ProcessingJob)
            .where(
                ProcessingJob.kind.in_(VECTOR_JOB_KINDS),
                ProcessingJob.status.in_(
                    [ProcessingJobStatus.PENDING, ProcessingJobStatus.RUNNING]
                ),
            )
            .order_by(ProcessingJob.created_at.asc())
            .limit(1)
        )


def _model_names_match(expected: str, actual: str) -> bool:
    expected_name = expected.strip().casefold()
    actual_name = actual.strip().casefold()
    if expected_name == actual_name:
        return True
    return (":" not in expected_name and actual_name == f"{expected_name}:latest") or (
        ":" not in actual_name and expected_name == f"{actual_name}:latest"
    )


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, OllamaError):
        return error.message[:2000]
    if isinstance(error, (VectorStoreError, VectorIndexError)):
        return error.message[:2000]
    return "Une erreur interne a interrompu l'indexation locale."
