from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from second_brain.api.dependencies import (
    get_app_settings,
    get_brain_runner,
    get_brain_service,
    get_session,
)
from second_brain.core.config import Settings
from second_brain.db.models.brain import (
    BrainCluster,
    BrainEdge,
    BrainNodeLayout,
    BrainProfile,
    BrainProfileStatus,
)
from second_brain.db.models.knowledge import KnowledgeNode
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    processing_job_is_stale,
)
from second_brain.db.models.taxonomy import KnowledgeNodeTag
from second_brain.jobs.brain_runner import BrainRunner
from second_brain.schemas.brain import (
    BrainClusterDetail,
    BrainClusterOut,
    BrainClusterSizeStats,
    BrainGraphEdge,
    BrainGraphNode,
    BrainGraphOut,
    BrainJobOut,
    BrainKnowledgeNodeOut,
    BrainProfileOut,
    BrainRebuildRequest,
    BrainRelabelRequest,
    BrainSearchAncestor,
    BrainSearchOut,
    BrainSearchResult,
    BrainSimilarityStats,
    BrainStatusOut,
)
from second_brain.services.brain import (
    BRAIN_JOB_KINDS,
    BrainBusyError,
    BrainError,
    BrainNotReadyError,
    BrainProfileNotFoundError,
    BrainService,
)

router = APIRouter(prefix="/brain", tags=["brain"])

MAX_CLUSTER_LIST = 500
MAX_GRAPH_NODES = 1_000
MAX_GRAPH_EDGES = 5_000
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_QUERY_LENGTH = 120
_SEARCH_WORD = re.compile(r"\w+", re.UNICODE)


@router.get("/status", response_model=BrainStatusOut)
async def brain_status(
    service: BrainService = Depends(get_brain_service),
    settings: Settings = Depends(get_app_settings),
) -> BrainStatusOut:
    snapshot = await service.status()
    return BrainStatusOut(
        state=snapshot.state,
        active_profile=_profile_out(snapshot.active_profile),
        building_profile=_profile_out(snapshot.building_profile),
        active_job=(
            _job_out(snapshot.active_job, settings) if snapshot.active_job is not None else None
        ),
        latest_job=(
            _job_out(snapshot.latest_job, settings) if snapshot.latest_job is not None else None
        ),
        stale_reasons=list(snapshot.stale_reasons),
        can_rebuild=snapshot.can_rebuild,
        can_relabel=snapshot.can_relabel,
        error=snapshot.error,
    )


@router.post(
    "/rebuild",
    response_model=BrainJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def rebuild_brain(
    payload: BrainRebuildRequest,
    runner: BrainRunner = Depends(get_brain_runner),
    settings: Settings = Depends(get_app_settings),
) -> BrainJobOut:
    del payload
    job = await _enqueue(runner, ProcessingJobKind.BUILD_BRAIN)
    return _job_out(job, settings)


@router.post(
    "/relabel",
    response_model=BrainJobOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def relabel_brain(
    payload: BrainRelabelRequest,
    runner: BrainRunner = Depends(get_brain_runner),
    settings: Settings = Depends(get_app_settings),
) -> BrainJobOut:
    del payload
    job = await _enqueue(runner, ProcessingJobKind.RELABEL_BRAIN)
    return _job_out(job, settings)


@router.get("/jobs/{job_id}", response_model=BrainJobOut)
async def brain_job(
    job_id: UUID,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_app_settings),
) -> BrainJobOut:
    job = await session.get(ProcessingJob, job_id)
    if job is None or job.kind not in BRAIN_JOB_KINDS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Traitement du cerveau introuvable.",
        )
    return _job_out(job, settings)


@router.get("/clusters", response_model=list[BrainClusterOut])
async def brain_clusters(
    level: int | None = Query(default=None, ge=0, le=32),
    parent_id: UUID | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=MAX_CLUSTER_LIST),
    session: AsyncSession = Depends(get_session),
) -> list[BrainClusterOut]:
    profile = await _require_usable_profile(session)
    statement = select(BrainCluster).where(BrainCluster.brain_profile_id == profile.id)
    if level is not None:
        statement = statement.where(BrainCluster.level == level)
    if parent_id is not None:
        statement = statement.where(BrainCluster.parent_cluster_id == parent_id)
    clusters = list(
        (
            await session.scalars(
                statement.order_by(
                    BrainCluster.level.asc(),
                    BrainCluster.member_count.desc(),
                    BrainCluster.label.asc(),
                    BrainCluster.id.asc(),
                ).limit(limit)
            )
        ).all()
    )
    child_counts = await _child_counts(session, profile.id, clusters)
    return [_cluster_out(cluster, child_counts.get(cluster.id, 0)) for cluster in clusters]


@router.get("/clusters/{cluster_id}", response_model=BrainClusterDetail)
async def brain_cluster_detail(
    cluster_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> BrainClusterDetail:
    profile = await _require_usable_profile(session)
    cluster = await _cluster_in_profile(session, profile.id, cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cluster du cerveau introuvable.",
        )
    children = list(
        (
            await session.scalars(
                select(BrainCluster)
                .where(
                    BrainCluster.brain_profile_id == profile.id,
                    BrainCluster.parent_cluster_id == cluster.id,
                )
                .order_by(
                    BrainCluster.member_count.desc(),
                    BrainCluster.label.asc(),
                    BrainCluster.id.asc(),
                )
            )
        ).all()
    )
    child_counts = await _child_counts(session, profile.id, [cluster, *children])
    all_clusters = list(
        (
            await session.scalars(
                select(BrainCluster).where(BrainCluster.brain_profile_id == profile.id)
            )
        ).all()
    )
    descendant_ids = _descendant_ids(cluster.id, all_clusters)
    layouts = await _load_layouts(
        session,
        profile.id,
        cluster_ids=descendant_ids,
    )
    return BrainClusterDetail(
        **_cluster_out(cluster, child_counts.get(cluster.id, 0)).model_dump(),
        children=[_cluster_out(child, child_counts.get(child.id, 0)) for child in children],
        knowledge_nodes=[_knowledge_out(layout) for layout in layouts],
    )


@router.get("/graph", response_model=BrainGraphOut)
async def brain_graph(
    level: int | None = Query(default=None, ge=0, le=32),
    cluster_id: UUID | None = Query(default=None),
    node_limit: int = Query(default=500, ge=1, le=MAX_GRAPH_NODES),
    edge_limit: int = Query(default=2_000, ge=0, le=MAX_GRAPH_EDGES),
    session: AsyncSession = Depends(get_session),
) -> BrainGraphOut:
    if level is not None and cluster_id is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Choisissez un niveau ou un cluster, pas les deux.",
        )
    profile = await _require_usable_profile(session)
    clusters = list(
        (
            await session.scalars(
                select(BrainCluster)
                .where(BrainCluster.brain_profile_id == profile.id)
                .order_by(BrainCluster.level.asc(), BrainCluster.id.asc())
            )
        ).all()
    )
    layouts = await _load_layouts(session, profile.id)
    edges = list(
        (
            await session.scalars(
                select(BrainEdge)
                .where(BrainEdge.brain_profile_id == profile.id)
                .order_by(BrainEdge.final_score.desc(), BrainEdge.id.asc())
            )
        ).all()
    )
    if cluster_id is not None:
        return _cluster_graph(
            profile=profile,
            clusters=clusters,
            layouts=layouts,
            edges=edges,
            cluster_id=cluster_id,
            node_limit=node_limit,
            edge_limit=edge_limit,
        )
    return _level_graph(
        profile=profile,
        clusters=clusters,
        layouts=layouts,
        edges=edges,
        requested_level=level,
        node_limit=node_limit,
        edge_limit=edge_limit,
    )


@router.get("/search", response_model=BrainSearchOut)
async def search_brain(
    q: str = Query(min_length=1, max_length=MAX_SEARCH_QUERY_LENGTH),
    limit: int = Query(default=20, ge=1, le=MAX_SEARCH_RESULTS),
    session: AsyncSession = Depends(get_session),
) -> BrainSearchOut:
    query = q.strip()
    normalized_query = _normalize_search_text(query)
    if not normalized_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="La recherche ne peut pas etre vide.",
        )

    profile = await _require_usable_profile(session)
    clusters = list(
        (
            await session.scalars(
                select(BrainCluster)
                .where(BrainCluster.brain_profile_id == profile.id)
                .order_by(BrainCluster.level.asc(), BrainCluster.id.asc())
            )
        ).all()
    )
    layouts = await _load_layouts(session, profile.id)
    clusters_by_id = {cluster.id: cluster for cluster in clusters}
    ranked_results: list[tuple[tuple[int, int], int, str, str, BrainSearchResult]] = []

    for cluster in clusters:
        search_rank = _search_rank(cluster.label, normalized_query)
        if search_rank is None:
            continue
        ranked_results.append(
            (
                search_rank,
                0,
                _normalize_search_text(cluster.label),
                str(cluster.id),
                BrainSearchResult(
                    kind="cluster",
                    target_id=cluster.id,
                    label=cluster.label,
                    level=cluster.level,
                    cluster_id=cluster.id,
                    x=cluster.x,
                    y=cluster.y,
                    member_count=cluster.member_count,
                    ancestors=_cluster_path(cluster.parent_cluster_id, clusters_by_id),
                ),
            )
        )

    for layout in layouts:
        node = layout.knowledge_node
        search_rank = _search_rank(node.title, normalized_query)
        if search_rank is None:
            continue
        leaf_cluster = (
            clusters_by_id.get(layout.cluster_id) if layout.cluster_id is not None else None
        )
        ranked_results.append(
            (
                search_rank,
                1,
                _normalize_search_text(node.title),
                str(node.id),
                BrainSearchResult(
                    kind="knowledge",
                    target_id=node.id,
                    label=node.title,
                    level=leaf_cluster.level + 1 if leaf_cluster is not None else None,
                    cluster_id=layout.cluster_id,
                    x=layout.x,
                    y=layout.y,
                    tags=sorted({link.tag.name for link in node.tag_links}),
                    source_id=node.source_id,
                    source_title=node.source.title,
                    href=f"/connaissances/{node.id}",
                    ancestors=_cluster_path(layout.cluster_id, clusters_by_id),
                ),
            )
        )

    ranked_results.sort(key=lambda result: result[:4])
    return BrainSearchOut(
        profile_id=profile.id,
        query=query,
        items=[result[-1] for result in ranked_results[:limit]],
    )


async def _enqueue(runner: BrainRunner, kind: ProcessingJobKind) -> ProcessingJob:
    try:
        return await runner.enqueue(kind)
    except BrainBusyError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    except (BrainNotReadyError, BrainProfileNotFoundError) as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error
    except BrainError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.message) from error


async def _require_usable_profile(session: AsyncSession) -> BrainProfile:
    for profile_status in (BrainProfileStatus.READY, BrainProfileStatus.STALE):
        profile = await session.scalar(
            select(BrainProfile)
            .where(BrainProfile.status == profile_status)
            .order_by(BrainProfile.logical_generation.desc())
            .limit(1)
        )
        if profile is not None:
            return profile
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Construisez le cerveau avant de consulter son graphe.",
    )


async def _cluster_in_profile(
    session: AsyncSession,
    profile_id: UUID,
    cluster_id: UUID,
) -> BrainCluster | None:
    return await session.scalar(
        select(BrainCluster).where(
            BrainCluster.brain_profile_id == profile_id,
            BrainCluster.id == cluster_id,
        )
    )


async def _child_counts(
    session: AsyncSession,
    profile_id: UUID,
    clusters: Iterable[BrainCluster],
) -> dict[UUID, int]:
    identifiers = [cluster.id for cluster in clusters]
    if not identifiers:
        return {}
    rows = (
        await session.execute(
            select(BrainCluster.parent_cluster_id, func.count(BrainCluster.id))
            .where(
                BrainCluster.brain_profile_id == profile_id,
                BrainCluster.parent_cluster_id.in_(identifiers),
            )
            .group_by(BrainCluster.parent_cluster_id)
        )
    ).all()
    return {parent_id: int(count) for parent_id, count in rows if parent_id is not None}


async def _load_layouts(
    session: AsyncSession,
    profile_id: UUID,
    *,
    cluster_ids: set[UUID] | None = None,
) -> list[BrainNodeLayout]:
    statement = (
        select(BrainNodeLayout)
        .where(BrainNodeLayout.brain_profile_id == profile_id)
        .options(
            selectinload(BrainNodeLayout.knowledge_node).selectinload(KnowledgeNode.source),
            selectinload(BrainNodeLayout.knowledge_node)
            .selectinload(KnowledgeNode.tag_links)
            .selectinload(KnowledgeNodeTag.tag),
        )
        .order_by(BrainNodeLayout.knowledge_node_id.asc())
    )
    if cluster_ids is not None:
        statement = statement.where(BrainNodeLayout.cluster_id.in_(cluster_ids))
    return list((await session.scalars(statement)).unique().all())


def _descendant_ids(cluster_id: UUID, clusters: Iterable[BrainCluster]) -> set[UUID]:
    children_by_parent: dict[UUID, list[UUID]] = {}
    for cluster in clusters:
        if cluster.parent_cluster_id is not None:
            children_by_parent.setdefault(cluster.parent_cluster_id, []).append(cluster.id)
    result: set[UUID] = set()
    pending = [cluster_id]
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(children_by_parent.get(current, ()))
    return result


def _profile_out(profile: BrainProfile | None) -> BrainProfileOut | None:
    if profile is None:
        return None
    statistics = _json_object(profile.statistics_json)
    levels = statistics.get("cluster_counts_by_level")
    similarity = statistics.get("similarity")
    cluster_sizes = statistics.get("cluster_sizes")
    return BrainProfileOut(
        id=profile.id,
        logical_generation=profile.logical_generation,
        status=profile.status.value,
        embedding_profile_id=profile.embedding_profile_id,
        embedding_provider=profile.embedding_provider,
        embedding_model_name=profile.embedding_model_name,
        embedding_model_digest=profile.embedding_model_digest,
        embedding_dimensions=profile.embedding_dimensions,
        embedding_semantic_text_version=profile.embedding_semantic_text_version,
        embedding_logical_generation=profile.embedding_logical_generation,
        algorithm_version=profile.algorithm_version,
        knowledge_node_count=profile.knowledge_node_count,
        cluster_count=profile.cluster_count,
        edge_count=profile.edge_count,
        unassigned_node_count=profile.unassigned_node_count,
        cluster_counts_by_level=_integer_mapping(levels),
        similarity=BrainSimilarityStats(**_numeric_stats(similarity)),
        cluster_sizes=BrainClusterSizeStats(**_numeric_stats(cluster_sizes)),
        relations_duration_ms=profile.relations_duration_ms,
        clustering_duration_ms=profile.clustering_duration_ms,
        umap_duration_ms=profile.umap_duration_ms,
        labeling_duration_ms=profile.labeling_duration_ms,
        total_duration_ms=profile.total_duration_ms,
        label_strategy=profile.label_strategy.value,
        label_model_name=profile.label_model_name,
        label_model_digest=profile.label_model_digest,
        created_at=profile.created_at,
        completed_at=profile.completed_at,
        activated_at=profile.activated_at,
        error_message=profile.error_message,
    )


def _job_out(job: ProcessingJob, settings: Settings) -> BrainJobOut:
    result = BrainJobOut.model_validate(job)
    return result.model_copy(
        update={
            "is_stale": processing_job_is_stale(
                job,
                stale_after_seconds=settings.job_stale_heartbeat_seconds,
            )
        }
    )


def _cluster_out(cluster: BrainCluster, child_count: int) -> BrainClusterOut:
    return BrainClusterOut(
        id=cluster.id,
        parent_id=cluster.parent_cluster_id,
        level=cluster.level,
        label=cluster.label,
        description=cluster.description,
        label_source=cluster.label_source.value,
        member_count=cluster.member_count,
        representative_knowledge_node_ids=_uuid_list(cluster.representative_nodes_json),
        x=cluster.x,
        y=cluster.y,
        child_count=child_count,
    )


def _knowledge_out(layout: BrainNodeLayout) -> BrainKnowledgeNodeOut:
    node = layout.knowledge_node
    return BrainKnowledgeNodeOut(
        id=node.id,
        cluster_id=layout.cluster_id,
        title=node.title,
        tags=sorted({link.tag.name for link in node.tag_links}),
        source_id=node.source_id,
        source_title=node.source.title,
        x=layout.x,
        y=layout.y,
        is_unassigned=layout.is_unassigned,
        href=f"/connaissances/{node.id}",
    )


def _cluster_graph(
    *,
    profile: BrainProfile,
    clusters: list[BrainCluster],
    layouts: list[BrainNodeLayout],
    edges: list[BrainEdge],
    cluster_id: UUID,
    node_limit: int,
    edge_limit: int,
) -> BrainGraphOut:
    clusters_by_id = {cluster.id: cluster for cluster in clusters}
    cluster = clusters_by_id.get(cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cluster du cerveau introuvable.",
        )
    children = sorted(
        (item for item in clusters if item.parent_cluster_id == cluster.id),
        key=lambda item: (-item.member_count, item.label.casefold(), str(item.id)),
    )
    if children:
        chosen = children[:node_limit]
        displayed_ids = {item.id for item in chosen}
        graph_nodes = [_cluster_graph_node(item) for item in chosen]
        membership = {
            layout.knowledge_node_id: f"cluster:{ancestor.id}"
            for layout in layouts
            if layout.cluster_id is not None
            and (
                ancestor := _descendant_branch(
                    layout.cluster_id,
                    parent_id=cluster.id,
                    clusters_by_id=clusters_by_id,
                )
            )
            is not None
            and ancestor.id in displayed_ids
        }
        graph_level = min(item.level for item in children)
        truncated = len(children) > len(chosen)
    else:
        members = sorted(
            (layout for layout in layouts if layout.cluster_id == cluster.id),
            key=lambda layout: (
                layout.knowledge_node.title.casefold(),
                str(layout.knowledge_node_id),
            ),
        )
        chosen_layouts = members[:node_limit]
        graph_nodes = [_knowledge_graph_node(layout) for layout in chosen_layouts]
        membership = {
            layout.knowledge_node_id: f"knowledge:{layout.knowledge_node_id}"
            for layout in chosen_layouts
        }
        graph_level = cluster.level + 1
        truncated = len(members) > len(chosen_layouts)
    graph_edges, edges_truncated = _aggregate_edges(edges, membership, edge_limit=edge_limit)
    return BrainGraphOut(
        profile_id=profile.id,
        level=graph_level,
        parent_cluster_id=cluster.id,
        nodes=graph_nodes,
        edges=graph_edges,
        truncated=truncated or edges_truncated,
    )


def _level_graph(
    *,
    profile: BrainProfile,
    clusters: list[BrainCluster],
    layouts: list[BrainNodeLayout],
    edges: list[BrainEdge],
    requested_level: int | None,
    node_limit: int,
    edge_limit: int,
) -> BrainGraphOut:
    available_levels = sorted({cluster.level for cluster in clusters})
    if requested_level is None:
        level = 1 if 1 in available_levels else (available_levels[0] if available_levels else 0)
    else:
        level = requested_level
    candidates = sorted(
        (cluster for cluster in clusters if cluster.level == level),
        key=lambda cluster: (-cluster.member_count, cluster.label.casefold(), str(cluster.id)),
    )
    chosen = candidates[:node_limit]
    remaining_slots = max(0, node_limit - len(chosen))
    unassigned = sorted(
        (layout for layout in layouts if layout.is_unassigned),
        key=lambda layout: (
            layout.knowledge_node.title.casefold(),
            str(layout.knowledge_node_id),
        ),
    )
    chosen_unassigned = unassigned[:remaining_slots]
    selected_ids = {cluster.id for cluster in chosen}
    graph_nodes = [
        *(_cluster_graph_node(cluster) for cluster in chosen),
        *(_knowledge_graph_node(layout) for layout in chosen_unassigned),
    ]
    clusters_by_id = {cluster.id: cluster for cluster in clusters}
    membership: dict[UUID, str] = {}
    for layout in layouts:
        if layout.cluster_id is None:
            continue
        ancestor = _ancestor_at_level(layout.cluster_id, level, clusters_by_id)
        if ancestor is not None and ancestor.id in selected_ids:
            membership[layout.knowledge_node_id] = f"cluster:{ancestor.id}"
    membership.update(
        {
            layout.knowledge_node_id: f"knowledge:{layout.knowledge_node_id}"
            for layout in chosen_unassigned
        }
    )
    graph_edges, edges_truncated = _aggregate_edges(edges, membership, edge_limit=edge_limit)
    return BrainGraphOut(
        profile_id=profile.id,
        level=level,
        parent_cluster_id=None,
        nodes=graph_nodes,
        edges=graph_edges,
        truncated=(
            len(candidates) + len(unassigned) > len(chosen) + len(chosen_unassigned)
            or edges_truncated
        ),
    )


def _ancestor_at_level(
    cluster_id: UUID,
    level: int,
    clusters_by_id: dict[UUID, BrainCluster],
) -> BrainCluster | None:
    current = clusters_by_id.get(cluster_id)
    visited: set[UUID] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        if current.level == level:
            return current
        if current.level < level or current.parent_cluster_id is None:
            return None
        current = clusters_by_id.get(current.parent_cluster_id)
    return None


def _descendant_branch(
    cluster_id: UUID,
    *,
    parent_id: UUID,
    clusters_by_id: dict[UUID, BrainCluster],
) -> BrainCluster | None:
    current = clusters_by_id.get(cluster_id)
    visited: set[UUID] = set()
    while current is not None and current.id not in visited:
        visited.add(current.id)
        if current.parent_cluster_id == parent_id:
            return current
        current = (
            clusters_by_id.get(current.parent_cluster_id)
            if current.parent_cluster_id is not None
            else None
        )
    return None


def _cluster_path(
    cluster_id: UUID | None,
    clusters_by_id: dict[UUID, BrainCluster],
) -> list[BrainSearchAncestor]:
    path: list[BrainCluster] = []
    visited: set[UUID] = set()
    current = clusters_by_id.get(cluster_id) if cluster_id is not None else None
    while current is not None and current.id not in visited:
        visited.add(current.id)
        path.append(current)
        current = (
            clusters_by_id.get(current.parent_cluster_id)
            if current.parent_cluster_id is not None
            else None
        )
    return [
        BrainSearchAncestor(id=cluster.id, label=cluster.label, level=cluster.level)
        for cluster in reversed(path)
    ]


def _normalize_search_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_accents.split())


def _search_rank(value: str, normalized_query: str) -> tuple[int, int] | None:
    candidate = _normalize_search_text(value)
    length_difference = max(0, len(candidate) - len(normalized_query))
    if candidate == normalized_query:
        return 0, length_difference
    if candidate.startswith(normalized_query):
        return 1, length_difference
    words = _SEARCH_WORD.findall(candidate)
    if any(word.startswith(normalized_query) for word in words):
        return 2, length_difference
    if normalized_query in candidate:
        return 3, length_difference
    query_words = _SEARCH_WORD.findall(normalized_query)
    if query_words and all(
        any(candidate_word.startswith(query_word) for candidate_word in words)
        for query_word in query_words
    ):
        return 4, length_difference
    return None


def _aggregate_edges(
    edges: Iterable[BrainEdge],
    membership: dict[UUID, str],
    *,
    edge_limit: int,
) -> tuple[list[BrainGraphEdge], bool]:
    aggregate: dict[tuple[str, str], tuple[float, int]] = {}
    for edge in edges:
        source = membership.get(edge.source_node_id)
        target = membership.get(edge.target_node_id)
        if source is None or target is None or source == target:
            continue
        pair = tuple(sorted((source, target)))
        previous_score, previous_count = aggregate.get(pair, (-1.0, 0))
        aggregate[pair] = (max(previous_score, edge.final_score), previous_count + 1)
    ordered = sorted(
        aggregate.items(),
        key=lambda item: (-item[1][0], str(item[0][0]), str(item[0][1])),
    )
    selected = ordered[:edge_limit]
    return (
        [
            BrainGraphEdge(
                source=source,
                target=target,
                score=score,
                relation_count=count,
            )
            for (source, target), (score, count) in selected
        ],
        len(ordered) > len(selected),
    )


def _cluster_graph_node(cluster: BrainCluster) -> BrainGraphNode:
    return BrainGraphNode(
        id=f"cluster:{cluster.id}",
        kind="cluster",
        label=cluster.label,
        x=cluster.x,
        y=cluster.y,
        size=cluster.member_count,
        cluster_id=cluster.id,
    )


def _knowledge_graph_node(layout: BrainNodeLayout) -> BrainGraphNode:
    node = layout.knowledge_node
    return BrainGraphNode(
        id=f"knowledge:{node.id}",
        kind="knowledge",
        label=node.title,
        x=layout.x,
        y=layout.y,
        size=1,
        cluster_id=layout.cluster_id,
        knowledge_node_id=node.id,
        source_id=node.source_id,
        tags=sorted({link.tag.name for link in node.tag_links}),
        href=f"/connaissances/{node.id}",
    )


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _integer_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    }


def _numeric_stats(value: object) -> dict[str, float | int | None]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float | int | None] = {}
    for key in ("minimum", "mean", "median", "maximum"):
        item = value.get(key)
        if item is None or isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = item
    return result


def _uuid_list(raw: str) -> list[UUID]:
    try:
        values = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    result: list[UUID] = []
    for value in values:
        try:
            result.append(UUID(str(value)))
        except (TypeError, ValueError):
            continue
    return result
