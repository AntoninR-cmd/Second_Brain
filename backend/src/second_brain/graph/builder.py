from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

import numpy as np

from second_brain.graph.clustering import build_cluster_hierarchy
from second_brain.graph.labels import apply_fallback_labels
from second_brain.graph.neighbors import build_knn_edges
from second_brain.graph.primitives import validated_node_matrix
from second_brain.graph.projection import project_to_2d
from second_brain.graph.types import (
    BrainMathConfig,
    BrainMathDurations,
    BrainMathNode,
    BrainMathResult,
    BrainMathStats,
    BrainNodeLayout,
    SimilarityDistribution,
)


def build_brain_math(
    nodes: Sequence[BrainMathNode],
    config: BrainMathConfig | None = None,
) -> BrainMathResult:
    """Construct the complete, reconstructible mathematical brain snapshot."""

    config = config or BrainMathConfig()
    started = perf_counter()
    ordered_nodes = tuple(sorted(nodes, key=lambda node: str(node.id)))
    vectors = validated_node_matrix(ordered_nodes)

    neighbors_started = perf_counter()
    edges = build_knn_edges(ordered_nodes, vectors, config)
    neighbors_seconds = perf_counter() - neighbors_started

    hierarchy = build_cluster_hierarchy(ordered_nodes, vectors, config)

    projection_started = perf_counter()
    projection = project_to_2d(vectors, config)
    projection_seconds = perf_counter() - projection_started
    positions = {
        node.id: (float(projection.coordinates[index, 0]), float(projection.coordinates[index, 1]))
        for index, node in enumerate(ordered_nodes)
    }
    positioned_clusters = tuple(
        replace(
            cluster,
            x=float(np.mean([positions[identifier][0] for identifier in cluster.member_ids])),
            y=float(np.mean([positions[identifier][1] for identifier in cluster.member_ids])),
        )
        for cluster in hierarchy.clusters
    )

    labeling_started = perf_counter()
    clusters = apply_fallback_labels(positioned_clusters, ordered_nodes)
    labeling_seconds = perf_counter() - labeling_started
    layouts = tuple(
        BrainNodeLayout(
            knowledge_node_id=node.id,
            cluster_id=hierarchy.deepest_cluster_by_node.get(node.id),
            x=positions[node.id][0],
            y=positions[node.id][1],
            unassigned=node.id in hierarchy.unassigned_ids,
        )
        for node in ordered_nodes
    )

    similarity = _similarity_distribution(edges)
    non_root_sizes = [cluster.size for cluster in clusters if cluster.level > 0]
    cluster_counts = dict(sorted(Counter(cluster.level for cluster in clusters).items()))
    durations = BrainMathDurations(
        neighbors_seconds=neighbors_seconds,
        pca_seconds=hierarchy.pca_seconds,
        clustering_seconds=hierarchy.clustering_seconds,
        projection_seconds=projection_seconds,
        labeling_seconds=labeling_seconds,
        total_seconds=perf_counter() - started,
    )
    stats = BrainMathStats(
        node_count=len(ordered_nodes),
        edge_count=len(edges),
        cluster_counts=cluster_counts,
        unassigned_count=len(hierarchy.unassigned_ids),
        cluster_size_min=min(non_root_sizes) if non_root_sizes else None,
        cluster_size_mean=float(np.mean(non_root_sizes)) if non_root_sizes else None,
        cluster_size_max=max(non_root_sizes) if non_root_sizes else None,
        similarity=similarity,
        durations=durations,
        projection_algorithm=projection.algorithm,
        pca_dimensions=hierarchy.pca_dimensions,
    )
    return BrainMathResult(nodes=layouts, edges=edges, clusters=clusters, stats=stats)


def _similarity_distribution(edges: Sequence[object]) -> SimilarityDistribution:
    scores = np.asarray([edge.cosine_score for edge in edges], dtype=np.float64)
    if not len(scores):
        return SimilarityDistribution()
    return SimilarityDistribution(
        minimum=float(scores.min()),
        mean=float(scores.mean()),
        median=float(np.median(scores)),
        maximum=float(scores.max()),
    )
