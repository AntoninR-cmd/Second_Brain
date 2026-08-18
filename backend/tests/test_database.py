from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from second_brain.db.session import Database
from sqlalchemy import text


def sqlite_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path.resolve().as_posix()}"


@pytest.mark.anyio
async def test_sqlite_uses_wal_and_foreign_keys(tmp_path: Path) -> None:
    database = Database(sqlite_url(tmp_path / "pragmas.sqlite3"))
    try:
        async with database.engine.connect() as connection:
            foreign_keys = await connection.scalar(text("PRAGMA foreign_keys"))
            journal_mode = await connection.scalar(text("PRAGMA journal_mode"))
    finally:
        await database.dispose()

    assert foreign_keys == 1
    assert str(journal_mode).lower() == "wal"


def test_alembic_cli_creates_a_missing_database_parent(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "missing" / "nested" / "migrated.sqlite3"
    environment = os.environ.copy()
    environment.update(
        {
            "SECOND_BRAIN_DATA_DIR": str(tmp_path / "data"),
            "SECOND_BRAIN_DATABASE_URL": sqlite_url(database_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert database_path.is_file()


def test_phase_two_migration_preserves_an_existing_manual_source(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "upgrade.sqlite3"
    environment = os.environ.copy()
    environment.update(
        {
            "SECOND_BRAIN_DATA_DIR": str(tmp_path / "data"),
            "SECOND_BRAIN_DATABASE_URL": sqlite_url(database_path),
        }
    )

    phase_one = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "20260816_0001"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert phase_one.returncode == 0, phase_one.stderr

    source_id = uuid4().hex
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO sources (
                id, type, title, author, raw_text, processing_status, created_at, updated_at
            ) VALUES (?, 'manual', 'Note Phase 1', NULL, 'Texte conservé', 'ready',
                      CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (source_id,),
        )

    phase_two = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert phase_two.returncode == 0, phase_two.stderr

    with sqlite3.connect(database_path) as connection:
        preserved = connection.execute(
            """
            SELECT id, type, title, raw_text, original_filename
            FROM sources
            """
        ).fetchone()
        segment_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'source_segments'"
        ).fetchone()
        source_ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sources'"
        ).fetchone()

    assert preserved == (source_id, "manual", "Note Phase 1", "Texte conservé", None)
    assert segment_table == ("source_segments",)
    assert source_ddl is not None
    assert "srt" in source_ddl[0]
    assert "txt" in source_ddl[0]
