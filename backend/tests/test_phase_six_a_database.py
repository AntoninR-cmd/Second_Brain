from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


def sqlite_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.resolve().as_posix()}"


def run_migration(
    repository_root: Path,
    database_path: Path,
    revision: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SECOND_BRAIN_DATA_DIR": str(database_path.parent / "data"),
            "SECOND_BRAIN_DATABASE_URL": sqlite_url(database_path),
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_phase_six_a_migration_creates_versioned_brain_schema(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "fresh-phase-six-a.sqlite3"

    migration = run_migration(repository_root, database_path, "head")
    assert migration.returncode == 0, migration.stderr

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        profile_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(brain_profiles)").fetchall()
        }
        cluster_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(brain_clusters)").fetchall()
        }
        layout_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(brain_node_layouts)").fetchall()
        }
        edge_columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(brain_edges)").fetchall()
        }
        job_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(processing_jobs)").fetchall()
        }
        job_ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'processing_jobs'"
        ).fetchone()
        ready_index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' "
            "AND name = 'uq_brain_profiles_single_ready'"
        ).fetchone()
        building_index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' "
            "AND name = 'uq_brain_profiles_single_building'"
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert {
        "brain_profiles",
        "brain_clusters",
        "brain_node_layouts",
        "brain_edges",
    }.issubset(tables)
    assert {
        "embedding_profile_id",
        "embedding_model_name",
        "embedding_model_digest",
        "embedding_dimensions",
        "embedding_semantic_text_version",
        "input_fingerprint",
        "algorithm_version",
        "parameters_json",
        "parameters_digest",
        "logical_generation",
        "status",
        "statistics_json",
        "relations_duration_ms",
        "clustering_duration_ms",
        "umap_duration_ms",
        "labeling_duration_ms",
        "total_duration_ms",
        "label_strategy",
        "activated_at",
    }.issubset(profile_columns)
    assert {
        "brain_profile_id",
        "parent_cluster_id",
        "level",
        "centroid_json",
        "representative_nodes_json",
        "member_count",
        "x",
        "y",
    }.issubset(cluster_columns)
    assert {
        "brain_profile_id",
        "knowledge_node_id",
        "cluster_id",
        "is_unassigned",
        "membership_confidence",
        "representative_rank",
        "x",
        "y",
    }.issubset(layout_columns)
    assert layout_columns["brain_profile_id"][5] == 1
    assert layout_columns["knowledge_node_id"][5] == 2
    assert {
        "brain_profile_id",
        "source_node_id",
        "target_node_id",
        "cosine_score",
        "tag_bonus",
        "final_score",
        "is_mutual",
    }.issubset(edge_columns)
    assert "brain_profile_id" in job_columns
    assert job_ddl is not None
    assert "build_brain" in job_ddl[0]
    assert "relabel_brain" in job_ddl[0]
    assert "kind != 'analyze_source' OR source_id IS NOT NULL" in job_ddl[0]
    assert "ck_processing_jobs_analysis_metrics" in job_ddl[0]
    assert "ck_processing_jobs_embedding_metrics" in job_ddl[0]
    assert ready_index is not None and "WHERE status = 'ready'" in ready_index[0]
    assert building_index is not None and "WHERE status = 'building'" in building_index[0]


def test_phase_six_a_upgrade_preserves_phase_five_data_and_job_constraints(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "phase-five-to-six-a.sqlite3"

    phase_five = run_migration(repository_root, database_path, "20260818_0006")
    assert phase_five.returncode == 0, phase_five.stderr

    source_id = uuid4().hex
    node_id = uuid4().hex
    embedding_profile_id = uuid4().hex
    analysis_job_id = uuid4().hex
    vector_job_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, raw_text, processing_status,
                analysis_status, created_at, updated_at
            ) VALUES (?, 'manual', 'Source Phase 5', 'Texte conserve', 'ready',
                      'analyzed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (source_id,),
        )
        connection.execute(
            """
            INSERT INTO knowledge_nodes (
                id, source_id, title, content, created_at, updated_at
            ) VALUES (?, ?, 'Connaissance preservee', 'Contenu autonome conserve',
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (node_id, source_id),
        )
        connection.execute(
            """
            INSERT INTO embedding_profiles (
                id, provider, model_name, model_digest, dimensions, distance,
                collection_name, semantic_text_version, logical_generation,
                status, created_at, updated_at, activated_at
            ) VALUES (?, 'ollama', 'qwen3-embedding:0.6b', 'digest-phase-five', 1024,
                      'cosine', 'second_brain_nodes_g1', 'knowledge_title_content_v1',
                      1, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (embedding_profile_id,),
        )
        connection.execute(
            """
            INSERT INTO knowledge_embeddings (
                knowledge_node_id, embedding_profile_id, text_fingerprint,
                status, indexed_at
            ) VALUES (?, ?, ?, 'indexed', CURRENT_TIMESTAMP)
            """,
            (node_id, embedding_profile_id, "a" * 64),
        )
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, source_id, kind, status, stage, progress_current, progress_total,
                progress_percent, progress_message, attempt_count, created_at, updated_at,
                last_activity_at
            ) VALUES (?, ?, 'analyze_source', 'succeeded', 'completed', 1, 1, 100,
                      'Analyse terminee.', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                      CURRENT_TIMESTAMP)
            """,
            (analysis_job_id, source_id),
        )
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, embedding_profile_id, kind, status, stage, progress_current,
                progress_total, progress_percent, progress_message, attempt_count,
                created_at, updated_at, last_activity_at
            ) VALUES (?, ?, 'index_knowledge', 'succeeded', 'completed', 1, 1, 100,
                      'Indexation terminee.', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                      CURRENT_TIMESTAMP)
            """,
            (vector_job_id, embedding_profile_id),
        )

    phase_six = run_migration(repository_root, database_path, "head")
    assert phase_six.returncode == 0, phase_six.stderr

    brain_profile_id = uuid4().hex
    brain_job_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        preserved_source = connection.execute(
            "SELECT id, title, raw_text, analysis_status FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()
        preserved_node = connection.execute(
            "SELECT id, source_id, title, content FROM knowledge_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        preserved_embedding = connection.execute(
            """
            SELECT knowledge_node_id, embedding_profile_id, text_fingerprint, status
            FROM knowledge_embeddings
            """
        ).fetchone()
        preserved_jobs = connection.execute(
            "SELECT id, kind, status FROM processing_jobs ORDER BY kind"
        ).fetchall()

        _insert_brain_profile(
            connection,
            profile_id=brain_profile_id,
            embedding_profile_id=embedding_profile_id,
            generation=1,
            status="ready",
        )
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, brain_profile_id, kind, status, stage, progress_current,
                progress_total, progress_percent, attempt_count, created_at,
                updated_at, last_activity_at
            ) VALUES (?, ?, 'build_brain', 'succeeded', 'completed', 1, 1, 100, 1,
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (brain_job_id, brain_profile_id),
        )
        brain_job = connection.execute(
            "SELECT brain_profile_id, kind, status FROM processing_jobs WHERE id = ?",
            (brain_job_id,),
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert preserved_source == (source_id, "Source Phase 5", "Texte conserve", "analyzed")
    assert preserved_node == (
        node_id,
        source_id,
        "Connaissance preservee",
        "Contenu autonome conserve",
    )
    assert preserved_embedding == (node_id, embedding_profile_id, "a" * 64, "indexed")
    assert preserved_jobs == [
        (analysis_job_id, "analyze_source", "succeeded"),
        (vector_job_id, "index_knowledge", "succeeded"),
    ]
    assert brain_job == (brain_profile_id, "build_brain", "succeeded")


def test_brain_schema_enforces_atomic_profiles_and_same_profile_graph(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "brain-integrity.sqlite3"
    migration = run_migration(repository_root, database_path, "head")
    assert migration.returncode == 0, migration.stderr

    source_id = uuid4().hex
    embedding_profile_id = uuid4().hex
    node_ids = sorted([uuid4().hex, uuid4().hex])
    ready_profile_id = uuid4().hex
    stale_profile_id = uuid4().hex
    ready_cluster_id = uuid4().hex
    stale_cluster_id = uuid4().hex

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, raw_text, processing_status,
                analysis_status, created_at, updated_at
            ) VALUES (?, 'manual', 'Source', 'Texte', 'ready', 'analyzed',
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (source_id,),
        )
        connection.executemany(
            """
            INSERT INTO knowledge_nodes (
                id, source_id, title, content, created_at, updated_at
            ) VALUES (?, ?, 'Noeud', 'Contenu', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            [(node_id, source_id) for node_id in node_ids],
        )
        connection.execute(
            """
            INSERT INTO embedding_profiles (
                id, provider, model_name, model_digest, dimensions, distance,
                collection_name, semantic_text_version, logical_generation,
                status, created_at, updated_at, activated_at
            ) VALUES (?, 'ollama', 'embedding-test', 'digest', 3, 'cosine',
                      'brain_test_vectors', 'semantic-v1', 1, 'active',
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (embedding_profile_id,),
        )
        _insert_brain_profile(
            connection,
            profile_id=ready_profile_id,
            embedding_profile_id=embedding_profile_id,
            generation=1,
            status="ready",
        )
        _insert_brain_profile(
            connection,
            profile_id=stale_profile_id,
            embedding_profile_id=embedding_profile_id,
            generation=2,
            status="stale",
        )

        with pytest.raises(sqlite3.IntegrityError):
            _insert_brain_profile(
                connection,
                profile_id=uuid4().hex,
                embedding_profile_id=embedding_profile_id,
                generation=3,
                status="ready",
            )

        _insert_brain_profile(
            connection,
            profile_id=uuid4().hex,
            embedding_profile_id=embedding_profile_id,
            generation=4,
            status="building",
        )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_brain_profile(
                connection,
                profile_id=uuid4().hex,
                embedding_profile_id=embedding_profile_id,
                generation=5,
                status="building",
            )

        _insert_cluster(connection, ready_cluster_id, ready_profile_id, parent_id=None)
        _insert_cluster(connection, stale_cluster_id, stale_profile_id, parent_id=None)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_cluster(
                connection,
                uuid4().hex,
                stale_profile_id,
                parent_id=ready_cluster_id,
            )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_layout(
                connection,
                profile_id=stale_profile_id,
                node_id=node_ids[0],
                cluster_id=ready_cluster_id,
            )

        # Cluster IDs are deterministic from their members. The same ID must be
        # reusable in two retained profile generations.
        _insert_cluster(connection, ready_cluster_id, stale_profile_id, parent_id=None)

        for node_id in node_ids:
            _insert_layout(
                connection,
                profile_id=ready_profile_id,
                node_id=node_id,
                cluster_id=ready_cluster_id,
            )
        connection.execute(
            """
            INSERT INTO brain_edges (
                id, brain_profile_id, source_node_id, target_node_id,
                cosine_score, tag_bonus, final_score, is_mutual
            ) VALUES (?, ?, ?, ?, 0.8, 0.02, 0.82, 1)
            """,
            (uuid4().hex, ready_profile_id, node_ids[0], node_ids[1]),
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO brain_edges (
                    id, brain_profile_id, source_node_id, target_node_id,
                    cosine_score, tag_bonus, final_score, is_mutual
                ) VALUES (?, ?, ?, ?, 0.8, 0.02, 0.82, 1)
                """,
                (uuid4().hex, ready_profile_id, node_ids[1], node_ids[0]),
            )

        connection.execute("DELETE FROM brain_profiles WHERE id = ?", (ready_profile_id,))
        derived_counts = tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("brain_edges", "brain_node_layouts")
        )
        business_counts = tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("sources", "knowledge_nodes", "embedding_profiles")
        )

    assert derived_counts == (0, 0)
    assert business_counts == (1, 2, 1)


def _insert_brain_profile(
    connection: sqlite3.Connection,
    *,
    profile_id: str,
    embedding_profile_id: str,
    generation: int,
    status: str,
) -> None:
    ready_date = "CURRENT_TIMESTAMP" if status == "ready" else "NULL"
    connection.execute(
        f"""
        INSERT INTO brain_profiles (
            id, embedding_profile_id, embedding_provider, embedding_model_name,
            embedding_model_digest, embedding_dimensions,
            embedding_semantic_text_version, embedding_logical_generation,
            input_fingerprint, algorithm_version, parameters_digest,
            logical_generation, status, completed_at, activated_at
        ) VALUES (?, ?, 'ollama', 'embedding-test', 'digest', 3, 'semantic-v1', 1,
                  ?, 'brain-math-v1', ?, ?, ?, {ready_date}, {ready_date})
        """,
        (profile_id, embedding_profile_id, "a" * 64, "b" * 64, generation, status),
    )


def _insert_cluster(
    connection: sqlite3.Connection,
    cluster_id: str,
    profile_id: str,
    *,
    parent_id: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO brain_clusters (
            id, brain_profile_id, parent_cluster_id, level, label, member_count,
            centroid_json, representative_nodes_json, x, y
        ) VALUES (?, ?, ?, 1, 'Cluster', 2, '[1.0, 0.0, 0.0]', '[]', 0.0, 0.0)
        """,
        (cluster_id, profile_id, parent_id),
    )


def _insert_layout(
    connection: sqlite3.Connection,
    *,
    profile_id: str,
    node_id: str,
    cluster_id: str,
) -> None:
    connection.execute(
        """
        INSERT INTO brain_node_layouts (
            brain_profile_id, knowledge_node_id, cluster_id, x, y,
            is_unassigned, membership_confidence
        ) VALUES (?, ?, ?, 0.0, 0.0, 0, 0.9)
        """,
        (profile_id, node_id, cluster_id),
    )
