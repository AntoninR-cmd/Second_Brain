from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
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
        populate_by_name=True,
    )

    app_name: str = "Second Brain"
    env: Literal["development", "test", "production"] = "development"
    data_dir: Path = Field(default_factory=default_data_directory)
    database_url: str | None = None
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_mb: int = Field(default=20, ge=1, le=500)
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "SECOND_BRAIN_OLLAMA_BASE_URL"),
    )
    ollama_generation_model: str = Field(
        default="qwen3.5:4b",
        validation_alias=AliasChoices(
            "OLLAMA_GENERATION_MODEL", "SECOND_BRAIN_OLLAMA_GENERATION_MODEL"
        ),
    )
    ollama_request_timeout_seconds: float = Field(
        default=600,
        gt=0,
        le=3600,
        validation_alias=AliasChoices(
            "OLLAMA_REQUEST_TIMEOUT_SECONDS",
            "SECOND_BRAIN_OLLAMA_REQUEST_TIMEOUT_SECONDS",
        ),
    )
    ollama_readiness_timeout_seconds: float = Field(
        default=5,
        gt=0,
        le=60,
        validation_alias=AliasChoices(
            "OLLAMA_READINESS_TIMEOUT_SECONDS",
            "SECOND_BRAIN_OLLAMA_READINESS_TIMEOUT_SECONDS",
        ),
    )
    ollama_num_ctx: int = Field(
        default=8192,
        ge=512,
        le=131_072,
        validation_alias=AliasChoices("OLLAMA_NUM_CTX", "SECOND_BRAIN_OLLAMA_NUM_CTX"),
    )
    ollama_temperature: float = Field(
        default=0.2,
        ge=0,
        le=2,
        validation_alias=AliasChoices("OLLAMA_TEMPERATURE", "SECOND_BRAIN_OLLAMA_TEMPERATURE"),
    )
    ollama_extraction_temperature: float = Field(
        default=0.0,
        ge=0,
        le=1,
        validation_alias=AliasChoices(
            "OLLAMA_EXTRACTION_TEMPERATURE",
            "SECOND_BRAIN_OLLAMA_EXTRACTION_TEMPERATURE",
        ),
    )
    ollama_keep_alive: str = Field(
        default="5m",
        min_length=1,
        max_length=32,
        validation_alias=AliasChoices("OLLAMA_KEEP_ALIVE", "SECOND_BRAIN_OLLAMA_KEEP_ALIVE"),
    )
    ollama_num_predict_passage_analysis: int = Field(
        default=512,
        ge=64,
        le=32_768,
        validation_alias=AliasChoices(
            "OLLAMA_NUM_PREDICT_PASSAGE_ANALYSIS",
            "SECOND_BRAIN_OLLAMA_NUM_PREDICT_PASSAGE_ANALYSIS",
        ),
    )
    ollama_num_predict_hierarchical_summary: int = Field(
        default=512,
        ge=64,
        le=32_768,
        validation_alias=AliasChoices(
            "OLLAMA_NUM_PREDICT_HIERARCHICAL_SUMMARY",
            "SECOND_BRAIN_OLLAMA_NUM_PREDICT_HIERARCHICAL_SUMMARY",
        ),
    )
    ollama_num_predict_final_summary: int = Field(
        default=1024,
        ge=64,
        le=32_768,
        validation_alias=AliasChoices(
            "OLLAMA_NUM_PREDICT_FINAL_SUMMARY",
            "SECOND_BRAIN_OLLAMA_NUM_PREDICT_FINAL_SUMMARY",
        ),
    )
    chunk_target_tokens: int = Field(
        default=800,
        ge=32,
        le=32_768,
        validation_alias=AliasChoices("CHUNK_TARGET_TOKENS", "SECOND_BRAIN_CHUNK_TARGET_TOKENS"),
    )
    chunk_max_tokens: int = Field(
        default=1200,
        ge=64,
        le=65_536,
        validation_alias=AliasChoices("CHUNK_MAX_TOKENS", "SECOND_BRAIN_CHUNK_MAX_TOKENS"),
    )
    chunk_overlap_segments: int = Field(
        default=2,
        ge=0,
        le=20,
        validation_alias=AliasChoices(
            "CHUNK_OVERLAP_SEGMENTS", "SECOND_BRAIN_CHUNK_OVERLAP_SEGMENTS"
        ),
    )
    chunk_srt_pause_ms: int = Field(
        default=2500,
        ge=0,
        le=120_000,
        validation_alias=AliasChoices("CHUNK_SRT_PAUSE_MS", "SECOND_BRAIN_CHUNK_SRT_PAUSE_MS"),
    )
    extraction_max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        validation_alias=AliasChoices(
            "EXTRACTION_MAX_RETRIES", "SECOND_BRAIN_EXTRACTION_MAX_RETRIES"
        ),
    )
    extraction_max_knowledge_per_passage: int = Field(
        default=2,
        ge=1,
        le=4,
        validation_alias=AliasChoices(
            "EXTRACTION_MAX_KNOWLEDGE_PER_PASSAGE",
            "SECOND_BRAIN_EXTRACTION_MAX_KNOWLEDGE_PER_PASSAGE",
        ),
    )
    job_stale_heartbeat_seconds: int = Field(
        default=120,
        ge=10,
        le=86_400,
        validation_alias=AliasChoices(
            "JOB_STALE_HEARTBEAT_SECONDS",
            "SECOND_BRAIN_JOB_STALE_HEARTBEAT_SECONDS",
        ),
    )

    @model_validator(mode="after")
    def validate_ai_configuration(self) -> Settings:
        self.ollama_base_url = self.ollama_base_url.strip().rstrip("/")
        self.ollama_generation_model = self.ollama_generation_model.strip()
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL doit utiliser http:// ou https://")
        if not self.ollama_generation_model:
            raise ValueError("OLLAMA_GENERATION_MODEL ne peut pas etre vide")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS doit etre inferieur ou egal a CHUNK_MAX_TOKENS")
        return self

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

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

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
