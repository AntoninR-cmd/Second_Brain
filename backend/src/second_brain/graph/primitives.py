from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from second_brain.graph.types import BrainMathNode

FloatMatrix = NDArray[np.float64]


def validated_node_matrix(nodes: Sequence[BrainMathNode]) -> FloatMatrix:
    if not nodes:
        return np.empty((0, 0), dtype=np.float64)
    if len({node.id for node in nodes}) != len(nodes):
        raise ValueError("les identifiants des KnowledgeNodes doivent etre uniques")
    dimension = len(nodes[0].vector)
    if dimension <= 0:
        raise ValueError("les embeddings doivent contenir au moins une dimension")
    if any(len(node.vector) != dimension for node in nodes):
        raise ValueError("tous les embeddings doivent avoir la meme dimension")
    matrix = np.asarray([node.vector for node in nodes], dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("les embeddings doivent contenir uniquement des nombres finis")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("un embedding ne peut pas etre nul")
    return matrix / norms[:, np.newaxis]


def cosine_matrix(normalized_vectors: FloatMatrix) -> FloatMatrix:
    if normalized_vectors.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    return np.clip(normalized_vectors @ normalized_vectors.T, -1.0, 1.0)


def normalized_centroid(vectors: FloatMatrix) -> NDArray[np.float64]:
    centroid = vectors.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= np.finfo(np.float64).eps:
        return np.zeros(vectors.shape[1], dtype=np.float64)
    return centroid / norm


def deterministic_pca(
    normalized_vectors: FloatMatrix,
    *,
    max_dimensions: int,
) -> tuple[FloatMatrix, int]:
    count, dimension = normalized_vectors.shape
    components = min(max_dimensions, max(0, count - 1), dimension)
    if components <= 0:
        return normalized_vectors.copy(), 0
    centered = normalized_vectors - normalized_vectors.mean(axis=0, keepdims=True)
    _left, _singular_values, right = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[:components].T
    norms = np.linalg.norm(projected, axis=1)
    valid = norms > np.finfo(np.float64).eps
    if not np.all(valid):
        # Cosine clustering cannot consume a zero row. This only happens for a
        # degenerate corpus (for example identical vectors), where retaining the
        # original normalized space is safer than inventing a direction.
        return normalized_vectors.copy(), 0
    projected[valid] /= norms[valid, np.newaxis]
    return projected, components


def tag_jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    left = {tag.strip().casefold() for tag in first if tag.strip()}
    right = {tag.strip().casefold() for tag in second if tag.strip()}
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)
