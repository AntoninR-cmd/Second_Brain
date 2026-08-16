from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
