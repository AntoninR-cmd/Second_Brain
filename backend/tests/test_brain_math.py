from __future__ import annotations

import sys
from collections import Counter
from types import ModuleType
from uuid import UUID, uuid5

import numpy as np
import pytest
from second_brain.graph import BrainMathConfig, BrainMathNode, build_brain_math
from second_brain.graph.projection import ProjectionResult

NAMESPACE = UUID("73d4024a-b7fd-4f27-aaca-81378da1df8c")


@pytest.fixture
def fast_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep non-projection unit tests independent from Numba warm-up time."""

    def project(vectors: np.ndarray, _config: BrainMathConfig) -> ProjectionResult:
        count = len(vectors)
        coordinates = np.column_stack((np.linspace(-1, 1, count), np.zeros(count)))
        return ProjectionResult(coordinates, "pca")

    monkeypatch.setattr("second_brain.graph.builder.project_to_2d", project)


def _node(
    name: str,
    vector: tuple[float, ...],
    *,
    tags: tuple[str, ...] = (),
) -> BrainMathNode:
    return BrainMathNode(
        id=uuid5(NAMESPACE, name),
        vector=vector,
        title=name.replace("-", " "),
        tags=tags,
    )


def _two_groups_with_noise() -> list[BrainMathNode]:
    return [
        _node("force-volume", (1.0, 0.04, 0.0, 0.0), tags=("musculation", "force")),
        _node("ordre-exercices", (0.99, -0.03, 0.01, 0.0), tags=("musculation",)),
        _node("fatigue", (0.98, 0.01, -0.02, 0.0), tags=("musculation",)),
        _node("couleurs", (0.01, 1.0, 0.03, 0.0), tags=("style", "couleurs")),
        _node("vetements", (-0.02, 0.99, -0.01, 0.0), tags=("style",)),
        _node("formalites", (0.03, 0.98, 0.02, 0.0), tags=("style",)),
        _node("cosmetiques-isole", (0.0, 0.0, 1.0, 0.0), tags=("cosmetiques",)),
    ]


def test_empty_corpus_builds_an_empty_reconstructible_result() -> None:
    result = build_brain_math([])

    assert result.nodes == ()
    assert result.edges == ()
    assert result.clusters == ()
    assert result.stats.node_count == 0
    assert result.stats.cluster_counts == {}
    assert result.stats.projection_algorithm == "empty"


def test_single_node_gets_a_stable_root_and_origin_position() -> None:
    node = _node("unique", (1.0, 0.0, 0.0))

    result = build_brain_math([node])

    assert len(result.nodes) == 1
    assert result.nodes[0].knowledge_node_id == node.id
    assert (result.nodes[0].x, result.nodes[0].y) == (0.0, 0.0)
    assert result.nodes[0].unassigned is False
    assert result.edges == ()
    assert len(result.clusters) == 1
    assert result.clusters[0].label == "Second Brain"
    assert result.nodes[0].cluster_id == result.clusters[0].id


def test_two_nodes_use_the_explicit_small_corpus_layout() -> None:
    nodes = [_node("a", (1.0, 0.0)), _node("b", (0.8, 0.6))]

    result = build_brain_math(nodes, BrainMathConfig(min_similarity=0.0))

    assert len(result.edges) == 1
    assert {(node.x, node.y) for node in result.nodes} == {(-1.0, 0.0), (1.0, 0.0)}
    assert result.stats.projection_algorithm == "two-node"


@pytest.mark.parametrize(
    "nodes, detail",
    [
        (
            [_node("same", (1.0, 0.0)), _node("same", (0.0, 1.0))],
            "uniques",
        ),
        ([_node("zero", (0.0, 0.0))], "nul"),
        (
            [_node("short", (1.0, 0.0)), _node("long", (1.0, 0.0, 0.0))],
            "meme dimension",
        ),
    ],
)
def test_invalid_vector_corpora_are_rejected(
    nodes: list[BrainMathNode],
    detail: str,
) -> None:
    with pytest.raises(ValueError, match=detail):
        build_brain_math(nodes)


def test_knn_edges_are_unique_canonical_and_degree_bounded(fast_projection: None) -> None:
    nodes = [_node(f"node-{index}", (1.0, index / 20, (index % 2) / 30)) for index in range(9)]
    config = BrainMathConfig(neighbors_k=3, min_similarity=0.0, min_cluster_size=2)

    result = build_brain_math(nodes, config)

    pairs = [(edge.source_node_id, edge.target_node_id) for edge in result.edges]
    assert len(pairs) == len(set(pairs))
    assert all(str(source) < str(target) for source, target in pairs)
    degrees = Counter(identifier for pair in pairs for identifier in pair)
    assert max(degrees.values()) <= 3
    assert any(edge.mutual for edge in result.edges)


def test_tag_bonus_is_a_small_jaccard_contribution() -> None:
    first = _node("first", (1.0, 0.0), tags=("bois", "vernis"))
    second = _node("second", (0.8, 0.6), tags=("bois", "peinture"))
    config = BrainMathConfig(min_similarity=0.0, tag_weight=0.1)

    edge = build_brain_math([first, second], config).edges[0]

    assert edge.cosine_score == pytest.approx(0.8)
    assert edge.tag_bonus == pytest.approx(0.1 / 3)
    assert edge.final_score == pytest.approx(0.9 * 0.8 + 0.1 / 3)


def test_two_semantic_groups_and_an_isolated_node_form_domains_and_noise(
    fast_projection: None,
) -> None:
    nodes = _two_groups_with_noise()
    config = BrainMathConfig(
        min_cluster_size=3,
        min_similarity=0.45,
        max_domain_clusters=3,
        max_theme_clusters=3,
    )

    result = build_brain_math(nodes, config)

    roots = [cluster for cluster in result.clusters if cluster.level == 0]
    domains = [cluster for cluster in result.clusters if cluster.level == 1]
    assert len(roots) == 1
    assert len(domains) == 2
    assert sorted(cluster.size for cluster in domains) == [3, 3]
    assert all(cluster.parent_id == roots[0].id for cluster in domains)
    isolated = next(node for node in result.nodes if node.knowledge_node_id == nodes[-1].id)
    assert isolated.unassigned is True
    assert isolated.cluster_id is None
    assert result.stats.unassigned_count == 1


def test_hierarchy_has_consistent_members_representatives_centroids_and_positions(
    fast_projection: None,
) -> None:
    nodes = _two_groups_with_noise()
    result = build_brain_math(
        nodes,
        BrainMathConfig(min_cluster_size=3, max_domain_clusters=3),
    )
    positions = {node.knowledge_node_id: (node.x, node.y) for node in result.nodes}

    for cluster in result.clusters:
        assert set(cluster.representative_ids) <= set(cluster.member_ids)
        assert len(cluster.centroid) == 4
        assert np.isfinite(cluster.centroid).all()
        assert cluster.x == pytest.approx(
            np.mean([positions[identifier][0] for identifier in cluster.member_ids])
        )
        assert cluster.y == pytest.approx(
            np.mean([positions[identifier][1] for identifier in cluster.member_ids])
        )
        assert cluster.label
    children_by_parent = {
        cluster.parent_id for cluster in result.clusters if cluster.parent_id is not None
    }
    assert children_by_parent <= {cluster.id for cluster in result.clusters}


def test_fixed_random_state_and_sorted_ids_make_umap_layout_and_structure_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = _two_groups_with_noise()
    config = BrainMathConfig(min_cluster_size=3, random_state=17)
    calls: list[dict[str, object]] = []

    class DeterministicUmap:
        def __init__(self, **parameters: object) -> None:
            self.parameters = parameters
            calls.append(parameters)

        def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
            generator = np.random.default_rng(int(self.parameters["random_state"]))
            return vectors[:, :2] + generator.normal(0, 1e-6, (len(vectors), 2))

    fake_umap = ModuleType("umap")
    fake_umap.UMAP = DeterministicUmap  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "umap", fake_umap)

    first = build_brain_math(nodes, config)
    second = build_brain_math(list(reversed(nodes)), config)

    assert first.stats.projection_algorithm == "umap"
    assert second.stats.projection_algorithm == "umap"
    assert all(call["random_state"] == 17 for call in calls)
    assert all(call["n_jobs"] == 1 for call in calls)
    assert [node.knowledge_node_id for node in first.nodes] == [
        node.knowledge_node_id for node in second.nodes
    ]
    assert first.edges == second.edges
    assert [cluster.id for cluster in first.clusters] == [cluster.id for cluster in second.clusters]
    assert np.allclose(
        [(node.x, node.y) for node in first.nodes],
        [(node.x, node.y) for node in second.nodes],
        atol=1e-6,
    )


def test_all_layout_coordinates_and_statistics_are_finite_and_bounded(
    fast_projection: None,
) -> None:
    result = build_brain_math(
        _two_groups_with_noise(),
        BrainMathConfig(min_cluster_size=3),
    )

    coordinates = np.asarray([(node.x, node.y) for node in result.nodes])
    assert np.isfinite(coordinates).all()
    assert np.max(np.abs(coordinates)) <= 1.0
    assert result.stats.node_count == 7
    assert result.stats.edge_count == len(result.edges)
    assert result.stats.cluster_counts[0] == 1
    assert result.stats.durations.total_seconds >= 0
    assert result.stats.durations.projection_seconds >= 0
