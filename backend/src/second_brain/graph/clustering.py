from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from second_brain.graph.primitives import (
    FloatMatrix,
    cosine_matrix,
    deterministic_pca,
    normalized_centroid,
)
from second_brain.graph.types import BrainCluster, BrainMathConfig, BrainMathNode

_CLUSTER_NAMESPACE = uuid5(NAMESPACE_URL, "second-brain/brain-math/clusters/v1")


@dataclass(frozen=True, slots=True)
class HierarchyResult:
    clusters: tuple[BrainCluster, ...]
    deepest_cluster_by_node: dict[UUID, UUID]
    unassigned_ids: frozenset[UUID]
    pca_dimensions: int
    pca_seconds: float
    clustering_seconds: float


def build_cluster_hierarchy(
    nodes: Sequence[BrainMathNode],
    normalized_vectors: FloatMatrix,
    config: BrainMathConfig,
) -> HierarchyResult:
    if not nodes:
        return HierarchyResult((), {}, frozenset(), 0, 0.0, 0.0)

    pca_started = perf_counter()
    prepared, pca_dimensions = deterministic_pca(
        normalized_vectors,
        max_dimensions=config.pca_dimensions,
    )
    pca_seconds = perf_counter() - pca_started
    clustering_started = perf_counter()

    root_members = tuple(range(len(nodes)))
    root = _make_cluster(
        nodes,
        normalized_vectors,
        root_members,
        parent_id=None,
        level=0,
        label="Second Brain",
        representative_count=config.representative_count,
    )
    if len(nodes) < 2:
        return HierarchyResult(
            clusters=(root,),
            deepest_cluster_by_node={nodes[0].id: root.id},
            unassigned_ids=frozenset(),
            pca_dimensions=pca_dimensions,
            pca_seconds=pca_seconds,
            clustering_seconds=perf_counter() - clustering_started,
        )

    isolated_indices = _isolated_indices(normalized_vectors, config)
    eligible = tuple(index for index in range(len(nodes)) if index not in isolated_indices)
    unassigned = frozenset(nodes[index].id for index in isolated_indices)
    deepest = {nodes[index].id: root.id for index in eligible}
    clusters: list[BrainCluster] = [root]

    domain_groups = _best_partition(
        prepared,
        eligible,
        max_clusters=config.max_domain_clusters,
        min_cluster_size=config.min_cluster_size,
        min_silhouette=config.min_silhouette,
    )
    if domain_groups is None:
        return HierarchyResult(
            clusters=tuple(clusters),
            deepest_cluster_by_node=deepest,
            unassigned_ids=unassigned,
            pca_dimensions=pca_dimensions,
            pca_seconds=pca_seconds,
            clustering_seconds=perf_counter() - clustering_started,
        )

    for domain_members in domain_groups:
        domain = _make_cluster(
            nodes,
            normalized_vectors,
            domain_members,
            parent_id=root.id,
            level=1,
            label="",
            representative_count=config.representative_count,
        )
        clusters.append(domain)
        for index in domain_members:
            deepest[nodes[index].id] = domain.id

        theme_groups = _best_partition(
            prepared,
            domain_members,
            max_clusters=config.max_theme_clusters,
            min_cluster_size=config.min_cluster_size,
            min_silhouette=config.min_silhouette,
        )
        if theme_groups is None:
            continue
        for theme_members in theme_groups:
            theme = _make_cluster(
                nodes,
                normalized_vectors,
                theme_members,
                parent_id=domain.id,
                level=2,
                label="",
                representative_count=config.representative_count,
            )
            clusters.append(theme)
            for index in theme_members:
                deepest[nodes[index].id] = theme.id

    clusters.sort(key=lambda cluster: (cluster.level, str(cluster.id)))
    return HierarchyResult(
        clusters=tuple(clusters),
        deepest_cluster_by_node=deepest,
        unassigned_ids=unassigned,
        pca_dimensions=pca_dimensions,
        pca_seconds=pca_seconds,
        clustering_seconds=perf_counter() - clustering_started,
    )


def _isolated_indices(vectors: FloatMatrix, config: BrainMathConfig) -> frozenset[int]:
    if len(vectors) < 3:
        return frozenset()
    similarities = cosine_matrix(vectors).copy()
    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.max(axis=1)
    first_quartile, third_quartile = np.quantile(nearest, [0.25, 0.75])
    robust_fence = first_quartile - config.isolation_iqr_multiplier * (
        third_quartile - first_quartile
    )
    # The measured graph threshold is a conservative ceiling: a dense corpus
    # must not turn every merely unusual node into noise.
    isolation_threshold = min(config.min_similarity, float(robust_fence))
    return frozenset(
        int(index) for index in np.flatnonzero(nearest < isolation_threshold - np.finfo(float).eps)
    )


def _best_partition(
    prepared_vectors: FloatMatrix,
    member_indices: Sequence[int],
    *,
    max_clusters: int,
    min_cluster_size: int,
    min_silhouette: float,
) -> tuple[tuple[int, ...], ...] | None:
    members = tuple(member_indices)
    maximum = min(max_clusters, len(members) // min_cluster_size)
    if maximum < 2:
        return None
    local_vectors = prepared_vectors[np.asarray(members)]
    local_similarities = cosine_matrix(local_vectors)
    if np.allclose(local_similarities, 1.0, atol=1e-10):
        return None

    best: tuple[float, int, tuple[tuple[int, ...], ...]] | None = None
    for cluster_count in range(2, maximum + 1):
        labels = _agglomerative_labels(local_vectors, cluster_count)
        local_groups = [np.flatnonzero(labels == label) for label in sorted(set(labels))]
        if any(len(group) < min_cluster_size for group in local_groups):
            continue
        score = _silhouette(local_similarities, labels)
        groups = tuple(
            sorted(
                (tuple(sorted(members[int(index)] for index in group)) for group in local_groups),
                key=lambda group: group,
            )
        )
        candidate = (score, -cluster_count, groups)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None or best[0] < min_silhouette:
        return None
    return best[2]


def _agglomerative_labels(vectors: FloatMatrix, cluster_count: int) -> np.ndarray:
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:  # pragma: no cover - production dependency, useful bootstrap fallback
        return _fallback_average_linkage(vectors, cluster_count)
    return AgglomerativeClustering(
        n_clusters=cluster_count,
        metric="cosine",
        linkage="average",
    ).fit_predict(vectors)


def _fallback_average_linkage(vectors: FloatMatrix, cluster_count: int) -> np.ndarray:
    distances = 1 - cosine_matrix(vectors)
    clusters: dict[int, tuple[int, ...]] = {index: (index,) for index in range(len(vectors))}
    next_id = len(vectors)
    while len(clusters) > cluster_count:
        keys = sorted(clusters)
        best: tuple[float, int, int] | None = None
        for offset, left in enumerate(keys):
            for right in keys[offset + 1 :]:
                distance = float(distances[np.ix_(clusters[left], clusters[right])].mean())
                candidate = (round(distance, 15), left, right)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _distance, left, right = best
        clusters[next_id] = tuple(sorted(clusters.pop(left) + clusters.pop(right)))
        next_id += 1
    labels = np.empty(len(vectors), dtype=int)
    for label, members in enumerate(sorted(clusters.values())):
        labels[np.asarray(members)] = label
    return labels


def _silhouette(similarities: FloatMatrix, labels: np.ndarray) -> float:
    distances = np.maximum(0.0, 1.0 - similarities)
    unique_labels = sorted(set(int(label) for label in labels))
    values: list[float] = []
    for index, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != index]
        if not len(same):
            values.append(0.0)
            continue
        within = float(distances[index, same].mean())
        nearest_other = min(
            float(distances[index, np.flatnonzero(labels == other)].mean())
            for other in unique_labels
            if other != label
        )
        denominator = max(within, nearest_other)
        values.append((nearest_other - within) / denominator if denominator else 0.0)
    return float(np.mean(values))


def _make_cluster(
    nodes: Sequence[BrainMathNode],
    vectors: FloatMatrix,
    member_indices: Sequence[int],
    *,
    parent_id: UUID | None,
    level: int,
    label: str,
    representative_count: int,
) -> BrainCluster:
    ordered_indices = tuple(sorted(member_indices, key=lambda index: str(nodes[index].id)))
    member_ids = tuple(nodes[index].id for index in ordered_indices)
    identifier = uuid5(
        _CLUSTER_NAMESPACE,
        f"level={level};members={','.join(str(identifier) for identifier in member_ids)}",
    )
    centroid = normalized_centroid(vectors[np.asarray(ordered_indices)])
    representative_indices = sorted(
        ordered_indices,
        key=lambda index: (-float(vectors[index] @ centroid), str(nodes[index].id)),
    )[:representative_count]
    return BrainCluster(
        id=identifier,
        parent_id=parent_id,
        level=level,
        member_ids=member_ids,
        centroid=tuple(float(value) for value in centroid),
        representative_ids=tuple(nodes[index].id for index in representative_indices),
        label=label,
        x=0.0,
        y=0.0,
    )
