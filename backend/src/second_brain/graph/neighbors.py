from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from second_brain.graph.primitives import FloatMatrix, tag_jaccard
from second_brain.graph.types import BrainEdge, BrainMathConfig, BrainMathNode


def build_knn_edges(
    nodes: Sequence[BrainMathNode],
    normalized_vectors: FloatMatrix,
    config: BrainMathConfig,
) -> tuple[BrainEdge, ...]:
    """Build a sparse, deterministic, undirected kNN graph.

    Mutual candidates are consumed first, then the remaining candidates by final
    score. The greedy degree guard keeps every node at or below ``neighbors_k``.
    """

    count = len(nodes)
    if count < 2:
        return ()
    effective_k = min(config.neighbors_k, count - 1)
    nearest = _nearest_neighbor_candidates(normalized_vectors, effective_k)
    directed_neighbors: list[set[int]] = []
    for index in range(count):
        candidates = sorted(
            nearest[index],
            key=lambda candidate: (-candidate[1], str(nodes[candidate[0]].id)),
        )
        directed_neighbors.append(
            {
                candidate_index
                for candidate_index, similarity in candidates[:effective_k]
                if similarity >= config.min_similarity
            }
        )

    candidates_by_pair: dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    for source_index, neighbors in enumerate(directed_neighbors):
        for target_index in neighbors:
            left, right = sorted((source_index, target_index))
            cosine = float(normalized_vectors[left] @ normalized_vectors[right])
            overlap = tag_jaccard(nodes[left].tags, nodes[right].tags)
            tag_contribution = config.tag_weight * overlap
            final_score = (1 - config.tag_weight) * cosine + tag_contribution
            candidates_by_pair[(left, right)] = (
                cosine,
                tag_contribution,
                float(np.clip(final_score, -1, 1)),
                left in directed_neighbors[right] and right in directed_neighbors[left],
            )

    ordered_candidates = sorted(
        candidates_by_pair.items(),
        key=lambda item: (
            not item[1][3],
            -item[1][2],
            -item[1][0],
            str(nodes[item[0][0]].id),
            str(nodes[item[0][1]].id),
        ),
    )
    degrees: defaultdict[int, int] = defaultdict(int)
    edges: list[BrainEdge] = []
    for (left, right), (cosine, tag_bonus, final_score, mutual) in ordered_candidates:
        if degrees[left] >= effective_k or degrees[right] >= effective_k:
            continue
        first, second = sorted((nodes[left].id, nodes[right].id), key=str)
        edges.append(
            BrainEdge(
                source_node_id=first,
                target_node_id=second,
                cosine_score=cosine,
                tag_bonus=tag_bonus,
                final_score=final_score,
                mutual=mutual,
            )
        )
        degrees[left] += 1
        degrees[right] += 1
    return tuple(
        sorted(edges, key=lambda edge: (str(edge.source_node_id), str(edge.target_node_id)))
    )


def _nearest_neighbor_candidates(
    vectors: FloatMatrix,
    neighbor_count: int,
) -> list[list[tuple[int, float]]]:
    """Return exact neighbors without materializing an n-by-n matrix."""

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:  # pragma: no cover - production dependency, useful bootstrap fallback
        similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
        result: list[list[tuple[int, float]]] = []
        for index in range(len(vectors)):
            result.append(
                [
                    (candidate, float(similarities[index, candidate]))
                    for candidate in range(len(vectors))
                    if candidate != index
                ]
            )
        return result

    model = NearestNeighbors(
        n_neighbors=min(len(vectors), neighbor_count + 1),
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    ).fit(vectors)
    distances, indices = model.kneighbors(vectors, return_distance=True)
    return [
        [
            (int(candidate), float(np.clip(1 - distance, -1, 1)))
            for candidate, distance in zip(row_indices, row_distances, strict=True)
            if int(candidate) != row_index
        ]
        for row_index, (row_indices, row_distances) in enumerate(
            zip(indices, distances, strict=True)
        )
    ]
