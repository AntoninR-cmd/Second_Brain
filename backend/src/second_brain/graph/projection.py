from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from second_brain.graph.primitives import FloatMatrix
from second_brain.graph.types import BrainMathConfig


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    coordinates: FloatMatrix
    algorithm: str


def project_to_2d(
    normalized_vectors: FloatMatrix,
    config: BrainMathConfig,
) -> ProjectionResult:
    count = len(normalized_vectors)
    if count == 0:
        return ProjectionResult(np.empty((0, 2), dtype=np.float64), "empty")
    if count == 1:
        return ProjectionResult(np.zeros((1, 2), dtype=np.float64), "single-node")
    if count == 2:
        return ProjectionResult(
            np.asarray(((-1.0, 0.0), (1.0, 0.0)), dtype=np.float64),
            "two-node",
        )
    if count == 3:
        return ProjectionResult(
            _normalized_coordinates(_pca_coordinates(normalized_vectors)), "pca"
        )

    try:
        from umap import UMAP
    except ImportError:  # pragma: no cover - production dependency, useful bootstrap fallback
        return ProjectionResult(
            _normalized_coordinates(_pca_coordinates(normalized_vectors)), "pca"
        )

    try:
        coordinates = UMAP(
            n_components=2,
            n_neighbors=min(config.umap_neighbors, count - 1),
            min_dist=config.umap_min_dist,
            metric="cosine",
            random_state=config.random_state,
            transform_seed=config.random_state,
            n_jobs=1,
        ).fit_transform(normalized_vectors)
    except (RuntimeError, TypeError, ValueError):
        # A degenerate tiny corpus must not make the reconstructible brain fail.
        return ProjectionResult(
            _normalized_coordinates(_pca_coordinates(normalized_vectors)), "pca"
        )
    return ProjectionResult(_normalized_coordinates(np.asarray(coordinates)), "umap")


def _pca_coordinates(vectors: FloatMatrix) -> FloatMatrix:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _left, _singular_values, right = np.linalg.svd(centered, full_matrices=False)
    component_count = min(2, len(right))
    coordinates = centered @ right[:component_count].T
    if component_count == 1:
        coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(vectors))))
    return np.asarray(coordinates, dtype=np.float64)


def _normalized_coordinates(coordinates: FloatMatrix) -> FloatMatrix:
    if not np.isfinite(coordinates).all():
        raise ValueError("la projection contient des coordonnees non finies")
    normalized = np.zeros_like(coordinates, dtype=np.float64)
    for axis in range(2):
        minimum = float(coordinates[:, axis].min())
        maximum = float(coordinates[:, axis].max())
        span = maximum - minimum
        if span <= np.finfo(np.float64).eps:
            continue
        midpoint = (maximum + minimum) / 2
        normalized[:, axis] = (coordinates[:, axis] - midpoint) / (span / 2)
    return np.clip(normalized, -1.0, 1.0)
