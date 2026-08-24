from pathlib import Path

import pytest
from second_brain.core.config import Settings


def test_allowed_origins_accept_csv() -> None:
    settings = Settings(
        _env_file=None,
        allowed_origins="http://localhost:5173, http://127.0.0.1:5173",
    )
    assert settings.allowed_origin_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_allowed_origins_accept_json() -> None:
    settings = Settings(
        _env_file=None,
        allowed_origins=('["http://localhost:5173", "http://127.0.0.1:5173"]'),
    )
    assert settings.allowed_origin_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_data_directory_expands_windows_environment_variable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SECOND_BRAIN_TEST_DATA", str(tmp_path))
    settings = Settings(
        _env_file=None,
        data_dir=Path("%SECOND_BRAIN_TEST_DATA%/notes"),
    )

    assert settings.resolved_data_dir == (tmp_path / "notes").resolve()


def test_custom_database_parent_is_created(tmp_path: Path) -> None:
    database_path = tmp_path / "custom" / "nested" / "brain.sqlite3"
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        database_url=f"sqlite+aiosqlite:///{database_path.as_posix()}",
    )

    settings.create_data_directory()

    assert database_path.parent.is_dir()


def test_qdrant_path_is_resolved_inside_data_directory(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        qdrant_path=Path("derived-vectors"),
    )

    settings.create_data_directory()

    assert settings.resolved_qdrant_path == (tmp_path / "data" / "derived-vectors").resolve()
    assert settings.resolved_qdrant_path.is_dir()


def test_qdrant_path_must_be_a_dedicated_child_of_data_directory(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        qdrant_path=Path("."),
    )

    with pytest.raises(ValueError, match="sous-dossier dedie"):
        settings.create_data_directory()


def test_qdrant_path_cannot_overlap_original_source_files(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        qdrant_path=Path("originals"),
    )

    with pytest.raises(ValueError, match="fichiers originaux"):
        settings.create_data_directory()


def test_sqlite_database_cannot_be_inside_reconstructible_qdrant_path(
    tmp_path: Path,
) -> None:
    qdrant_path = tmp_path / "data" / "qdrant"
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        qdrant_path=qdrant_path,
        database_url=(f"sqlite+aiosqlite:///{(qdrant_path / 'second_brain.sqlite3').as_posix()}"),
    )

    with pytest.raises(ValueError, match="SQLite"):
        settings.create_data_directory()
