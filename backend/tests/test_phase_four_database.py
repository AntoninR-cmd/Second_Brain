from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


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


def test_phase_four_migration_creates_vector_index_schema(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "fresh-phase-four.sqlite3"

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
            for row in connection.execute("PRAGMA table_info(embedding_profiles)").fetchall()
        }
        embedding_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(knowledge_embeddings)").fetchall()
        }
        job_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(processing_jobs)").fetchall()
        }
        job_ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'processing_jobs'"
        ).fetchone()
        active_index = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' "
            "AND name = 'uq_embedding_profiles_single_active'"
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert {"embedding_profiles", "knowledge_embeddings"}.issubset(tables)
    assert {
        "provider",
        "model_name",
        "model_digest",
        "dimensions",
        "distance",
        "collection_name",
        "semantic_text_version",
        "logical_generation",
        "status",
        "activated_at",
    }.issubset(profile_columns)
    assert {
        "knowledge_node_id",
        "embedding_profile_id",
        "text_fingerprint",
        "status",
        "attempt_count",
        "indexed_at",
    }.issubset(embedding_columns)
    assert embedding_columns["knowledge_node_id"][5] == 1
    assert embedding_columns["embedding_profile_id"][5] == 2
    assert {
        "embedding_profile_id",
        "embedding_batch_count",
        "embedding_item_count",
        "embedding_duration_ms",
        "embedding_total_duration_ns",
        "embedding_prompt_eval_count",
    }.issubset(job_columns)
    assert job_columns["source_id"][3] == 0
    assert job_columns["embedding_batch_count"][4] == "0"
    assert job_ddl is not None
    assert "index_knowledge" in job_ddl[0]
    assert "rebuild_vector_index" in job_ddl[0]
    assert "kind != 'analyze_source' OR source_id IS NOT NULL" in job_ddl[0]
    assert active_index is not None
    assert "WHERE status = 'active'" in active_index[0]


def test_phase_four_upgrade_preserves_phase_three_business_data(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "phase-three-to-four.sqlite3"

    phase_three = run_migration(repository_root, database_path, "20260818_0005")
    assert phase_three.returncode == 0, phase_three.stderr

    source_id = uuid4().hex
    node_id = uuid4().hex
    analysis_job_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, raw_text, processing_status,
                analysis_status, created_at, updated_at
            ) VALUES (?, 'manual', 'Source Phase 3', 'Texte conserve', 'ready',
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

    phase_four = run_migration(repository_root, database_path, "head")
    assert phase_four.returncode == 0, phase_four.stderr

    profile_id = uuid4().hex
    vector_job_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        preserved_source = connection.execute(
            "SELECT id, title, raw_text, analysis_status FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()
        preserved_node = connection.execute(
            "SELECT id, source_id, title, content FROM knowledge_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        preserved_job = connection.execute(
            "SELECT id, source_id, kind, status FROM processing_jobs WHERE id = ?",
            (analysis_job_id,),
        ).fetchone()

        connection.execute(
            """
            INSERT INTO embedding_profiles (
                id, model_name, collection_name, semantic_text_version,
                logical_generation, status
            ) VALUES (?, 'qwen3-embedding:0.6b', 'second_brain_nodes_1',
                      'knowledge_title_content_v1', 1, 'building')
            """,
            (profile_id,),
        )
        connection.execute(
            """
            INSERT INTO knowledge_embeddings (
                knowledge_node_id, embedding_profile_id, text_fingerprint, status
            ) VALUES (?, ?, ?, 'pending')
            """,
            (node_id, profile_id, "a" * 64),
        )
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, source_id, embedding_profile_id, kind, status, stage,
                progress_current, progress_total, progress_percent,
                attempt_count, created_at, updated_at, last_activity_at
            ) VALUES (?, NULL, ?, 'index_knowledge', 'pending', 'queued',
                      0, 1, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (vector_job_id, profile_id),
        )
        vector_job = connection.execute(
            """
            SELECT source_id, embedding_profile_id, kind,
                   embedding_batch_count, embedding_item_count,
                   embedding_duration_ms, embedding_total_duration_ns,
                   embedding_prompt_eval_count
            FROM processing_jobs WHERE id = ?
            """,
            (vector_job_id,),
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert preserved_source == (source_id, "Source Phase 3", "Texte conserve", "analyzed")
    assert preserved_node == (
        node_id,
        source_id,
        "Connaissance preservee",
        "Contenu autonome conserve",
    )
    assert preserved_job == (analysis_job_id, source_id, "analyze_source", "succeeded")
    assert vector_job == (None, profile_id, "index_knowledge", 0, 0, 0, 0, 0)
