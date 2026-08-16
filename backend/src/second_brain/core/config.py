from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def default_data_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "SecondBrain" / "data"
    return REPOSITORY_ROOT / "data"


class Settings(BaseSettings):
    """Configuration loaded from ``SECOND_BRAIN_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_file=REPOSITORY_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="SECOND_BRAIN_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Second Brain"
    env: Literal["development", "test", "production"] = "development"
    data_dir: Path = Field(default_factory=default_data_directory)
    database_url: str | None = None
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def resolved_data_dir(self) -> Path:
        expanded_path = os.path.expandvars(str(self.data_dir))
        data_directory = Path(expanded_path).expanduser()
        if not data_directory.is_absolute():
            data_directory = REPOSITORY_ROOT / data_directory
        return data_directory.resolve()

    @property
    def resolved_database_url(self) -> str:
        configured_url = os.path.expandvars((self.database_url or "").strip())
        if configured_url:
            url = make_url(configured_url)
            if url.drivername.startswith("sqlite") and url.database and url.database != ":memory:":
                database_path = Path(url.database).expanduser()
                if not database_path.is_absolute():
                    database_path = REPOSITORY_ROOT / database_path
                url = url.set(database=database_path.resolve().as_posix())
            return url.render_as_string(hide_password=False)

        database_path = self.resolved_data_dir / "second_brain.sqlite3"
        return f"sqlite+aiosqlite:///{database_path.as_posix()}"

    @property
    def allowed_origin_list(self) -> list[str]:
        raw_origins = self.allowed_origins.strip()
        if raw_origins.startswith("["):
            try:
                decoded = json.loads(raw_origins)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list) and all(isinstance(origin, str) for origin in decoded):
                origins = (origin.strip() for origin in decoded)
                return list(dict.fromkeys(origin for origin in origins if origin))

        origins = (origin.strip() for origin in raw_origins.split(","))
        return list(dict.fromkeys(origin for origin in origins if origin))

    def create_data_directory(self) -> None:
        self.resolved_data_dir.mkdir(parents=True, exist_ok=True)

        database_url = make_url(self.resolved_database_url)
        if (
            database_url.drivername.startswith("sqlite")
            and database_url.database
            and database_url.database != ":memory:"
        ):
            Path(database_url.database).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
