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


def test_phase_three_migrations_create_the_complete_schema(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "fresh-phase-three.sqlite3"

    migration = run_migration(repository_root, database_path, "head")
    assert migration.returncode == 0, migration.stderr

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        source_columns = {
            row[1]: row for row in connection.execute("PRAGMA table_info(sources)").fetchall()
        }
        source_ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sources'"
        ).fetchone()
        job_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(processing_jobs)").fetchall()
        }
        job_ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'processing_jobs'"
        ).fetchone()
        passage_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(source_passages)").fetchall()
        }
        evidence_columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(knowledge_evidence)").fetchall()
        }
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert {
        "sources",
        "source_segments",
        "source_passages",
        "source_passage_segments",
        "knowledge_nodes",
        "knowledge_evidence",
        "tags",
        "knowledge_node_tags",
        "processing_jobs",
    }.issubset(tables)
    assert {
        "summary",
        "analysis_status",
        "analysis_error",
        "analysis_started_at",
        "analysis_completed_at",
    }.issubset(source_columns)
    assert source_columns["analysis_status"][4] == "'not_analyzed'"
    assert source_ddl is not None
    assert "not_analyzed" in source_ddl[0]
    assert "analyzed" in source_ddl[0]
    assert {
        "progress_percent",
        "last_activity_at",
        "error_code",
        "error_type",
        "error_detail",
        "error_stage",
        "error_passage_id",
        "error_passage_index",
        "error_attempt",
        "error_call_type",
        "llm_call_count",
        "knowledge_node_count",
    }.issubset(job_columns)
    assert job_columns["progress_percent"][3] == 1
    assert job_columns["progress_percent"][4] == "0"
    assert job_columns["last_activity_at"][3] == 1
    assert job_ddl is not None
    assert "progress_percent >= 0 AND progress_percent <= 100" in job_ddl[0]
    assert {
        "analysis_status",
        "analysis_payload_json",
        "analysis_error",
        "analysis_attempt_count",
        "analysis_last_activity_at",
        "llm_call_count",
        "knowledge_count",
    }.issubset(passage_columns)
    assert passage_columns["analysis_status"][4] == "'pending'"
    assert evidence_columns["passage_id"][3] == 1


def test_phase_three_upgrade_preserves_phase_two_source_and_segments(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "phase-two-upgrade.sqlite3"

    phase_two = run_migration(repository_root, database_path, "20260817_0002")
    assert phase_two.returncode == 0, phase_two.stderr

    source_id = uuid4().hex
    first_segment_id = uuid4().hex
    second_segment_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, author, original_filename, original_file_path,
                file_sha256, raw_text, processing_status, created_at, updated_at
            ) VALUES (?, 'srt', 'Source Phase 2', 'Auteur', 'episode.srt',
                      'originals/source/original.srt', ?, 'Premier\n\nSecond', 'ready',
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (source_id, "a" * 64),
        )
        connection.executemany(
            """
            INSERT INTO source_segments (
                id, source_id, segment_index, text, start_ms, end_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (first_segment_id, source_id, 1, "Premier", 1_000, 2_500),
                (second_segment_id, source_id, 2, "Second", 3_000, 4_250),
            ],
        )

    phase_three = run_migration(repository_root, database_path, "head")
    assert phase_three.returncode == 0, phase_three.stderr

    with sqlite3.connect(database_path) as connection:
        source = connection.execute(
            """
            SELECT id, type, title, author, original_filename, original_file_path,
                   file_sha256, raw_text, processing_status, summary, analysis_status,
                   analysis_error, analysis_started_at, analysis_completed_at
            FROM sources
            WHERE id = ?
            """,
            (source_id,),
        ).fetchone()
        segments = connection.execute(
            """
            SELECT id, source_id, segment_index, text, start_ms, end_ms
            FROM source_segments
            WHERE source_id = ?
            ORDER BY segment_index
            """,
            (source_id,),
        ).fetchall()

    assert source == (
        source_id,
        "srt",
        "Source Phase 2",
        "Auteur",
        "episode.srt",
        "originals/source/original.srt",
        "a" * 64,
        "Premier\n\nSecond",
        "ready",
        None,
        "not_analyzed",
        None,
        None,
        None,
    )
    assert segments == [
        (first_segment_id, source_id, 1, "Premier", 1_000, 2_500),
        (second_segment_id, source_id, 2, "Second", 3_000, 4_250),
    ]


def test_progress_migration_preserves_existing_phase_three_job(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "phase-three-progress-upgrade.sqlite3"

    phase_three = run_migration(repository_root, database_path, "20260817_0003")
    assert phase_three.returncode == 0, phase_three.stderr

    source_id = uuid4().hex
    job_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, raw_text, processing_status,
                analysis_status, created_at, updated_at
            ) VALUES (?, 'manual', 'Source existante', 'Texte conservé', 'ready',
                      'processing', '2026-08-17 09:00:00', '2026-08-17 09:01:00')
            """,
            (source_id,),
        )
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, source_id, kind, status, stage, progress_current, progress_total,
                progress_message, error_message, attempt_count, created_at, updated_at,
                started_at, finished_at
            ) VALUES (?, ?, 'analyze_source', 'running', 'passages', 4, 12,
                      'Analyse des passages.', NULL, 1,
                      '2026-08-17 09:00:00', '2026-08-17 09:01:00',
                      '2026-08-17 09:00:30', NULL)
            """,
            (job_id, source_id),
        )

    progress_upgrade = run_migration(repository_root, database_path, "head")
    assert progress_upgrade.returncode == 0, progress_upgrade.stderr

    with sqlite3.connect(database_path) as connection:
        job = connection.execute(
            """
            SELECT id, source_id, kind, status, stage, progress_current, progress_total,
                   progress_message, error_message, attempt_count, created_at, updated_at,
                   started_at, finished_at, progress_percent, last_activity_at
            FROM processing_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert job == (
        job_id,
        source_id,
        "analyze_source",
        "running",
        "passages",
        4,
        12,
        "Analyse des passages.",
        None,
        1,
        "2026-08-17 09:00:00",
        "2026-08-17 09:01:00",
        "2026-08-17 09:00:30",
        None,
        0,
        "2026-08-17 09:01:00",
    )


def test_resumable_analysis_migration_preserves_phase_three_passages_and_job(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "phase-three-resume-upgrade.sqlite3"

    phase_three = run_migration(repository_root, database_path, "20260817_0004")
    assert phase_three.returncode == 0, phase_three.stderr

    source_id = uuid4().hex
    passage_id = uuid4().hex
    job_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, raw_text, processing_status,
                analysis_status, created_at, updated_at
            ) VALUES (?, 'manual', 'Source a reprendre', 'Texte conserve', 'ready',
                      'error', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (source_id,),
        )
        connection.execute(
            """
            INSERT INTO source_passages (
                id, source_id, passage_index, text, token_count,
                char_start, char_end, intermediate_summary
            ) VALUES (?, ?, 0, 'Texte conserve', 4, 0, 14,
                      'Resume intermediaire ancien')
            """,
            (passage_id, source_id),
        )
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, source_id, kind, status, stage, progress_current, progress_total,
                progress_percent, progress_message, attempt_count, created_at, updated_at,
                last_activity_at
            ) VALUES (?, ?, 'analyze_source', 'failed', 'failed', 1, 3, 28,
                      'Analyse interrompue.', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                      CURRENT_TIMESTAMP)
            """,
            (job_id, source_id),
        )

    resume_upgrade = run_migration(repository_root, database_path, "head")
    assert resume_upgrade.returncode == 0, resume_upgrade.stderr

    with sqlite3.connect(database_path) as connection:
        passage = connection.execute(
            """
            SELECT id, source_id, passage_index, text, intermediate_summary,
                   analysis_status, analysis_payload_json, analysis_error,
                   analysis_attempt_count, llm_call_count, knowledge_count
            FROM source_passages
            WHERE id = ?
            """,
            (passage_id,),
        ).fetchone()
        job = connection.execute(
            """
            SELECT id, status, stage, progress_current, progress_total, progress_percent,
                   error_code, error_type, error_detail, error_passage_id,
                   llm_call_count, knowledge_node_count
            FROM processing_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        evidence_passage = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(knowledge_evidence)").fetchall()
        }["passage_id"]
        revision = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert revision == ("20260824_0007",)
    assert passage == (
        passage_id,
        source_id,
        0,
        "Texte conserve",
        "Resume intermediaire ancien",
        "pending",
        None,
        None,
        0,
        0,
        0,
    )
    assert job == (
        job_id,
        "failed",
        "failed",
        1,
        3,
        28,
        None,
        None,
        None,
        None,
        0,
        0,
    )
    assert evidence_passage[3] == 1
