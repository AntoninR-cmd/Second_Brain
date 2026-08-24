from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from second_brain.api.dependencies import (
    get_app_settings,
    get_brain_runner,
    get_brain_service,
    get_session,
)
from second_brain.api.routes.brain import router
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
from second_brain.db.models.knowledge import KnowledgeNode
from second_brain.db.models.processing import (
    ProcessingJob,
    ProcessingJobKind,
    ProcessingJobStatus,
)
from second_brain.db.models.source import Source, SourceType
from second_brain.db.models.taxonomy import KnowledgeNodeTag, Tag
from second_brain.db.session import Database
from second_brain.services.brain import BrainBusyError, BrainStatusSnapshot


@dataclass(frozen=True)
class SeededBrain:
    profile: BrainProfile
    root_id: UUID
    first_cluster_id: UUID
    second_cluster_id: UUID
    first_theme_id: UUID
    first_node_id: UUID
    second_node_id: UUID
    build_job: ProcessingJob
    relabel_job: ProcessingJob


class FakeBrainService:
    def __init__(self, seeded: SeededBrain) -> None:
        self.seeded = seeded

    async def status(self) -> BrainStatusSnapshot:
        return BrainStatusSnapshot(
            state="ready",
            active_profile=self.seeded.profile,
            building_profile=None,
            active_job=None,
            latest_job=self.seeded.build_job,
            stale_reasons=(),
            can_rebuild=True,
            can_relabel=True,
        )


class FakeBrainRunner:
    def __init__(self, seeded: SeededBrain) -> None:
        self.seeded = seeded
        self.kinds: list[ProcessingJobKind] = []

    async def enqueue(self, kind: ProcessingJobKind) -> ProcessingJob:
        self.kinds.append(kind)
        if kind == ProcessingJobKind.BUILD_BRAIN:
            return self.seeded.build_job
        return self.seeded.relabel_job


class BusyBrainRunner:
    async def enqueue(self, kind: ProcessingJobKind) -> ProcessingJob:
        del kind
        raise BrainBusyError("Une construction est deja en cours.")


@pytest.fixture(scope="module")
def brain_api(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[TestClient, SeededBrain, FakeBrainRunner, Database]]:
    data_dir = tmp_path_factory.mktemp("phase-six-a-api")
    settings = Settings(
        _env_file=None,
        env="test",
        data_dir=data_dir,
        database_url=_sqlite_url(data_dir / "second-brain.sqlite3"),
    )
    asyncio.run(migrate_database(settings.database_url))
    database = Database(settings.database_url)
    seeded = asyncio.run(_seed_brain(database))
    service = FakeBrainService(seeded)
    runner = FakeBrainRunner(seeded)
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")

    async def session_override() -> AsyncIterator[object]:
        async with database.session_factory() as session:
            yield session

    application.dependency_overrides[get_session] = session_override
    application.dependency_overrides[get_app_settings] = lambda: settings
    application.dependency_overrides[get_brain_service] = lambda: service
    application.dependency_overrides[get_brain_runner] = lambda: runner
    with TestClient(application) as client:
        yield client, seeded, runner, database
    asyncio.run(database.dispose())


def _sqlite_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.resolve().as_posix()}"


def test_brain_status_and_actions_expose_versioned_profile(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, runner, _ = brain_api

    response = client.get("/api/v1/brain/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] == "ready"
    assert payload["active_profile"]["id"] == str(seeded.profile.id)
    assert payload["active_profile"]["cluster_counts_by_level"] == {
        "0": 1,
        "1": 2,
        "2": 1,
    }
    assert payload["active_profile"]["similarity"]["mean"] == 0.64
    assert payload["active_profile"]["cluster_sizes"]["maximum"] == 4
    assert "parameters_json" not in json.dumps(payload)
    assert "centroid" not in json.dumps(payload)

    assert client.post("/api/v1/brain/rebuild", json={"confirm": False}).status_code == 422
    rebuilt = client.post("/api/v1/brain/rebuild", json={"confirm": True})
    relabeled = client.post("/api/v1/brain/relabel", json={"confirm": True})
    assert rebuilt.status_code == 202
    assert rebuilt.json()["kind"] == "build_brain"
    assert relabeled.status_code == 202
    assert relabeled.json()["kind"] == "relabel_brain"
    assert runner.kinds == [ProcessingJobKind.BUILD_BRAIN, ProcessingJobKind.RELABEL_BRAIN]
    polled = client.get(f"/api/v1/brain/jobs/{seeded.build_job.id}")
    assert polled.status_code == 200
    assert polled.json()["progress_percent"] == 100


def test_brain_action_reports_a_conflict_without_hiding_the_reason(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, _, runner, _ = brain_api
    client.app.dependency_overrides[get_brain_runner] = BusyBrainRunner

    response = client.post("/api/v1/brain/rebuild", json={"confirm": True})
    client.app.dependency_overrides[get_brain_runner] = lambda: runner

    assert response.status_code == 409
    assert response.json()["detail"] == "Une construction est deja en cours."


def test_brain_clusters_and_leaf_detail_reload_sqlite_knowledge(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, _, _ = brain_api

    roots = client.get("/api/v1/brain/clusters", params={"level": 0})
    children = client.get(
        "/api/v1/brain/clusters",
        params={"parent_id": str(seeded.root_id)},
    )
    detail = client.get(f"/api/v1/brain/clusters/{seeded.first_cluster_id}")

    assert roots.status_code == 200
    assert roots.json()[0]["child_count"] == 2
    assert {item["id"] for item in children.json()} == {
        str(seeded.first_cluster_id),
        str(seeded.second_cluster_id),
    }
    assert detail.status_code == 200
    payload = detail.json()
    assert [item["id"] for item in payload["children"]] == [str(seeded.first_theme_id)]
    assert {item["title"] for item in payload["knowledge_nodes"]} == {
        "Adherence du plastique",
        "Degraissage du support",
    }
    assert all(
        item["source_title"] == "Peindre un pare-chocs" for item in payload["knowledge_nodes"]
    )
    assert all(item["href"].startswith("/connaissances/") for item in payload["knowledge_nodes"])
    assert any("plastique" in item["tags"] for item in payload["knowledge_nodes"])
    assert "content" not in json.dumps(payload)


def test_brain_graph_aggregates_cross_cluster_edges_and_keeps_leaf_edges(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, _, _ = brain_api

    global_response = client.get("/api/v1/brain/graph", params={"level": 1})
    leaf_response = client.get(
        "/api/v1/brain/graph",
        params={"cluster_id": str(seeded.first_theme_id)},
    )

    assert global_response.status_code == 200
    global_graph = global_response.json()
    assert {node["kind"] for node in global_graph["nodes"]} == {"cluster"}
    assert len(global_graph["edges"]) == 1
    assert global_graph["edges"][0]["relation_count"] == 2
    assert global_graph["edges"][0]["score"] == 0.82
    assert all(edge["source"].startswith("cluster:") for edge in global_graph["edges"])

    assert leaf_response.status_code == 200
    leaf_graph = leaf_response.json()
    assert {node["kind"] for node in leaf_graph["nodes"]} == {"knowledge"}
    assert len(leaf_graph["edges"]) == 1
    assert leaf_graph["edges"][0]["score"] == 0.67
    assert leaf_graph["edges"][0]["source"].startswith("knowledge:")


def test_brain_api_rejects_ambiguous_or_unknown_graph_requests(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, _, _ = brain_api

    ambiguous = client.get(
        "/api/v1/brain/graph",
        params={"level": 1, "cluster_id": str(seeded.first_cluster_id)},
    )
    unknown = client.get(f"/api/v1/brain/clusters/{uuid4()}")
    unrelated_job = client.get(f"/api/v1/brain/jobs/{uuid4()}")

    assert ambiguous.status_code == 422
    assert unknown.status_code == 404
    assert unrelated_job.status_code == 404


def test_brain_search_returns_sqlite_knowledge_with_an_ordered_cluster_path(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, _, _ = brain_api

    response = client.get(
        "/api/v1/brain/search",
        params={"q": "  Adhérence du plastique  "},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile_id"] == str(seeded.profile.id)
    assert payload["query"] == "Adhérence du plastique"
    assert len(payload["items"]) == 1
    result = payload["items"][0]
    assert result == {
        "kind": "knowledge",
        "target_id": str(seeded.first_node_id),
        "label": "Adherence du plastique",
        "level": 3,
        "cluster_id": str(seeded.first_theme_id),
        "x": -0.6,
        "y": 0.05,
        "member_count": None,
        "tags": ["plastique"],
        "source_id": result["source_id"],
        "source_title": "Peindre un pare-chocs",
        "href": f"/connaissances/{seeded.first_node_id}",
        "ancestors": [
            {
                "id": str(seeded.root_id),
                "label": "Second Brain",
                "level": 0,
            },
            {
                "id": str(seeded.first_cluster_id),
                "label": "Preparation des plastiques",
                "level": 1,
            },
            {
                "id": str(seeded.first_theme_id),
                "label": "Adherence et degraissage",
                "level": 2,
            },
        ],
    }
    serialized = json.dumps(payload)
    assert "Un promoteur ameliore" not in serialized
    assert "centroid" not in serialized
    assert "vector" not in serialized


def test_brain_search_returns_cluster_ancestors_and_a_stable_bounded_order(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, _, _ = brain_api

    cluster_response = client.get(
        "/api/v1/brain/search",
        params={"q": "adhérence et dégraissage"},
    )
    first = client.get("/api/v1/brain/search", params={"q": "du", "limit": 1})
    second = client.get("/api/v1/brain/search", params={"q": "du", "limit": 1})

    assert cluster_response.status_code == 200
    cluster = cluster_response.json()["items"][0]
    assert cluster["kind"] == "cluster"
    assert cluster["target_id"] == str(seeded.first_theme_id)
    assert cluster["cluster_id"] == str(seeded.first_theme_id)
    assert cluster["member_count"] == 2
    assert [ancestor["id"] for ancestor in cluster["ancestors"]] == [
        str(seeded.root_id),
        str(seeded.first_cluster_id),
    ]

    assert first.status_code == 200
    assert first.json() == second.json()
    assert len(first.json()["items"]) == 1
    assert first.json()["items"][0]["target_id"] == str(seeded.first_node_id)


@pytest.mark.parametrize(
    ("params", "expected_status"),
    [
        ({"q": "   "}, 422),
        ({"q": "x" * 121}, 422),
        ({"q": "plastique", "limit": 0}, 422),
        ({"q": "plastique", "limit": 51}, 422),
    ],
)
def test_brain_search_validates_query_and_limit(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
    params: dict[str, str | int],
    expected_status: int,
) -> None:
    client, _, _, _ = brain_api

    response = client.get("/api/v1/brain/search", params=params)

    assert response.status_code == expected_status


def test_brain_search_returns_an_empty_list_for_no_local_match(
    brain_api: tuple[TestClient, SeededBrain, FakeBrainRunner, Database],
) -> None:
    client, seeded, _, _ = brain_api

    response = client.get("/api/v1/brain/search", params={"q": "ornithorynque"})

    assert response.status_code == 200
    assert response.json() == {
        "profile_id": str(seeded.profile.id),
        "query": "ornithorynque",
        "items": [],
    }


async def _seed_brain(database: Database) -> SeededBrain:
    now = utc_now()
    profile = BrainProfile(
        id=uuid4(),
        embedding_provider="ollama",
        embedding_model_name="qwen3-embedding:0.6b",
        embedding_model_digest="digest-test",
        embedding_dimensions=1024,
        embedding_semantic_text_version="knowledge_title_content_v1",
        embedding_logical_generation=1,
        input_fingerprint="a" * 64,
        algorithm_version="brain-math-v1",
        parameters_json="{}",
        parameters_digest="b" * 64,
        logical_generation=1,
        status=BrainProfileStatus.READY,
        knowledge_node_count=4,
        cluster_count=4,
        edge_count=3,
        unassigned_node_count=0,
        statistics_json=json.dumps(
            {
                "cluster_counts_by_level": {"0": 1, "1": 2, "2": 1},
                "cluster_sizes": {"minimum": 2, "mean": 2.67, "maximum": 4},
                "similarity": {
                    "minimum": 0.52,
                    "mean": 0.64,
                    "median": 0.63,
                    "maximum": 0.82,
                },
            }
        ),
        relations_duration_ms=12,
        clustering_duration_ms=18,
        umap_duration_ms=25,
        labeling_duration_ms=4,
        total_duration_ms=59,
        label_strategy=BrainLabelStrategy.DETERMINISTIC,
        completed_at=now,
        activated_at=now,
    )
    root_id, first_cluster_id, second_cluster_id, first_theme_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    root = _cluster(profile.id, root_id, None, 0, "Second Brain", 4, 0.0, 0.0)
    first_cluster = _cluster(
        profile.id,
        first_cluster_id,
        root_id,
        1,
        "Preparation des plastiques",
        2,
        -0.5,
        0.1,
    )
    second_cluster = _cluster(
        profile.id,
        second_cluster_id,
        root_id,
        1,
        "Recuperation sportive",
        2,
        0.5,
        -0.1,
    )
    first_theme = _cluster(
        profile.id,
        first_theme_id,
        first_cluster_id,
        2,
        "Adherence et degraissage",
        2,
        -0.5,
        0.1,
    )
    first_source = Source(
        type=SourceType.MANUAL,
        title="Peindre un pare-chocs",
        raw_text="Preparation, degraissage et adherence.",
    )
    second_source = Source(
        type=SourceType.MANUAL,
        title="Gerer la recuperation",
        raw_text="Fatigue, repos et volume.",
    )
    first_nodes = [
        KnowledgeNode(
            source=first_source,
            title="Adherence du plastique",
            content="Un promoteur ameliore l'adherence.",
        ),
        KnowledgeNode(
            source=first_source,
            title="Degraissage du support",
            content="Le support doit etre degraisse avant peinture.",
        ),
    ]
    second_nodes = [
        KnowledgeNode(
            source=second_source,
            title="Repos entre les seances",
            content="Le repos limite l'accumulation de fatigue.",
        ),
        KnowledgeNode(
            source=second_source,
            title="Ajuster le volume",
            content="Le volume baisse quand la recuperation se degrade.",
        ),
    ]
    tag = Tag(name="plastique", normalized_name="plastique")
    first_nodes[0].tag_links.append(KnowledgeNodeTag(tag=tag))
    layouts = [
        *(
            BrainNodeLayout(
                brain_profile_id=profile.id,
                knowledge_node=node,
                cluster_id=first_theme.id,
                x=-0.6 + index * 0.2,
                y=0.05 + index * 0.1,
                is_unassigned=False,
            )
            for index, node in enumerate(first_nodes)
        ),
        *(
            BrainNodeLayout(
                brain_profile_id=profile.id,
                knowledge_node=node,
                cluster_id=second_cluster.id,
                x=0.4 + index * 0.2,
                y=-0.15 + index * 0.1,
                is_unassigned=False,
            )
            for index, node in enumerate(second_nodes)
        ),
    ]
    build_job = ProcessingJob(
        brain_profile=profile,
        kind=ProcessingJobKind.BUILD_BRAIN,
        status=ProcessingJobStatus.SUCCEEDED,
        stage="completed",
        progress_current=5,
        progress_total=5,
        progress_percent=100,
        progress_message="Cerveau construit.",
        started_at=now,
        finished_at=now,
    )
    relabel_job = ProcessingJob(
        brain_profile=profile,
        kind=ProcessingJobKind.RELABEL_BRAIN,
        status=ProcessingJobStatus.SUCCEEDED,
        stage="completed",
        progress_current=2,
        progress_total=2,
        progress_percent=100,
        progress_message="Clusters renommes.",
        started_at=now,
        finished_at=now,
    )
    async with database.session_factory() as session:
        session.add_all(
            [
                profile,
                root,
                first_cluster,
                second_cluster,
                first_theme,
                first_source,
                second_source,
                *layouts,
                build_job,
                relabel_job,
            ]
        )
        await session.flush()
        session.add_all(
            [
                _edge(profile.id, first_nodes[0].id, first_nodes[1].id, 0.67),
                _edge(profile.id, first_nodes[0].id, second_nodes[0].id, 0.82),
                _edge(profile.id, first_nodes[1].id, second_nodes[1].id, 0.74),
            ]
        )
        await session.commit()
    return SeededBrain(
        profile=profile,
        root_id=root_id,
        first_cluster_id=first_cluster_id,
        second_cluster_id=second_cluster_id,
        first_theme_id=first_theme_id,
        first_node_id=first_nodes[0].id,
        second_node_id=first_nodes[1].id,
        build_job=build_job,
        relabel_job=relabel_job,
    )


def _cluster(
    profile_id: UUID,
    cluster_id: UUID,
    parent_id: UUID | None,
    level: int,
    label: str,
    member_count: int,
    x: float,
    y: float,
) -> BrainCluster:
    return BrainCluster(
        brain_profile_id=profile_id,
        id=cluster_id,
        parent_cluster_id=parent_id,
        level=level,
        label=label,
        label_source=BrainLabelSource.DETERMINISTIC,
        member_count=member_count,
        centroid_json="[1.0,0.0,0.0]",
        representative_nodes_json="[]",
        x=x,
        y=y,
    )


def _edge(profile_id: UUID, left: UUID, right: UUID, score: float) -> BrainEdge:
    source_id, target_id = sorted((left, right), key=str)
    return BrainEdge(
        brain_profile_id=profile_id,
        source_node_id=source_id,
        target_node_id=target_id,
        cosine_score=score,
        tag_bonus=0.0,
        final_score=score,
        is_mutual=True,
    )
