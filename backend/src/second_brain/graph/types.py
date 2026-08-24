from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True, slots=True)
class BrainMathNode:
    """One semantic node consumed by the pure graph builder."""

    id: UUID
    vector: tuple[float, ...]
    title: str
    tags: tuple[str, ...] = ()
    source_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class BrainMathConfig:
    neighbors_k: int = 8
    min_similarity: float = 0.45
    tag_weight: float = 0.05
    pca_dimensions: int = 50
    min_cluster_size: int = 5
    max_domain_clusters: int = 6
    max_theme_clusters: int = 6
    min_silhouette: float = 0.1
    isolation_iqr_multiplier: float = 1.5
    representative_count: int = 5
    umap_neighbors: int = 15
    umap_min_dist: float = 0.15
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.neighbors_k <= 0:
            raise ValueError("neighbors_k doit etre strictement positif")
        if not -1 <= self.min_similarity <= 1:
            raise ValueError("min_similarity doit etre compris entre -1 et 1")
        if not 0 <= self.tag_weight <= 1:
            raise ValueError("tag_weight doit etre compris entre 0 et 1")
        if self.pca_dimensions <= 0:
            raise ValueError("pca_dimensions doit etre strictement positif")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size doit etre au moins egal a 2")
        if self.max_domain_clusters < 2 or self.max_theme_clusters < 2:
            raise ValueError("les nombres maximum de clusters doivent etre au moins egaux a 2")
        if not -1 <= self.min_silhouette <= 1:
            raise ValueError("min_silhouette doit etre compris entre -1 et 1")
        if self.isolation_iqr_multiplier < 0:
            raise ValueError("isolation_iqr_multiplier doit etre positif")
        if self.representative_count <= 0:
            raise ValueError("representative_count doit etre strictement positif")
        if self.umap_neighbors < 2:
            raise ValueError("umap_neighbors doit etre au moins egal a 2")
        if not 0 <= self.umap_min_dist <= 1:
            raise ValueError("umap_min_dist doit etre compris entre 0 et 1")


@dataclass(frozen=True, slots=True)
class BrainEdge:
    source_node_id: UUID
    target_node_id: UUID
    cosine_score: float
    tag_bonus: float
    final_score: float
    mutual: bool


@dataclass(frozen=True, slots=True)
class BrainCluster:
    id: UUID
    parent_id: UUID | None
    level: int
    member_ids: tuple[UUID, ...]
    centroid: tuple[float, ...]
    representative_ids: tuple[UUID, ...]
    label: str
    x: float
    y: float

    @property
    def size(self) -> int:
        return len(self.member_ids)


@dataclass(frozen=True, slots=True)
class BrainNodeLayout:
    knowledge_node_id: UUID
    cluster_id: UUID | None
    x: float
    y: float
    unassigned: bool = False


@dataclass(frozen=True, slots=True)
class SimilarityDistribution:
    minimum: float | None = None
    mean: float | None = None
    median: float | None = None
    maximum: float | None = None


@dataclass(frozen=True, slots=True)
class BrainMathDurations:
    neighbors_seconds: float = 0.0
    pca_seconds: float = 0.0
    clustering_seconds: float = 0.0
    projection_seconds: float = 0.0
    labeling_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class BrainMathStats:
    node_count: int
    edge_count: int
    cluster_counts: dict[int, int]
    unassigned_count: int
    cluster_size_min: int | None
    cluster_size_mean: float | None
    cluster_size_max: int | None
    similarity: SimilarityDistribution
    durations: BrainMathDurations
    projection_algorithm: str
    pca_dimensions: int


@dataclass(frozen=True, slots=True)
class BrainMathResult:
    nodes: tuple[BrainNodeLayout, ...] = ()
    edges: tuple[BrainEdge, ...] = ()
    clusters: tuple[BrainCluster, ...] = ()
    stats: BrainMathStats = field(
        default_factory=lambda: BrainMathStats(
            node_count=0,
            edge_count=0,
            cluster_counts={},
            unassigned_count=0,
            cluster_size_min=None,
            cluster_size_mean=None,
            cluster_size_max=None,
            similarity=SimilarityDistribution(),
            durations=BrainMathDurations(),
            projection_algorithm="empty",
            pca_dimensions=0,
        )
    )
