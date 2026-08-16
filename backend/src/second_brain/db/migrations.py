from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _upgrade_database(database_url: str) -> None:
    config = Config(str(REPOSITORY_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")


async def migrate_database(database_url: str) -> None:
    """Upgrade the configured database without blocking FastAPI's event loop."""

    await asyncio.to_thread(_upgrade_database, database_url)
