from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from second_brain.core.config import get_settings
from second_brain.db.base import Base
from second_brain.db.models import Source  # noqa: F401
from second_brain.db.session import apply_sqlite_pragmas
from sqlalchemy import event, pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

config = context.config

if (
    config.config_file_name is not None
    and config.get_section("loggers")
    and config.attributes.get("configure_logger", True)
):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def database_url() -> str:
    configured = config.attributes.get("database_url")
    if isinstance(configured, str) and configured:
        return configured

    settings = get_settings()
    settings.create_data_directory()
    return settings.resolved_database_url


def run_migrations_offline() -> None:
    context.configure(
        url=database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        {"sqlalchemy.url": database_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    event.listen(connectable.sync_engine, "connect", apply_sqlite_pragmas)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
