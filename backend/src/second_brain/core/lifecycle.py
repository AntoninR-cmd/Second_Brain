from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from second_brain.db.migrations import migrate_database
from second_brain.db.session import Database


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = app.state.settings
    settings.create_data_directory()

    await migrate_database(settings.resolved_database_url)
    database = Database(settings.resolved_database_url)
    app.state.database = database

    try:
        yield
    finally:
        await database.dispose()
