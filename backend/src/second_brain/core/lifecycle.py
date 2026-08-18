from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from second_brain.core.logging import configure_application_logging
from second_brain.db.migrations import migrate_database
from second_brain.db.session import Database
from second_brain.jobs.analysis_runner import AnalysisRunner
from second_brain.llm.client import OllamaTextGenerator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_application_logging()
    settings = app.state.settings
    settings.create_data_directory()

    await migrate_database(settings.resolved_database_url)
    database = Database(settings.resolved_database_url)
    app.state.database = database
    generator = app.state.text_generator
    if generator is None:
        generator = OllamaTextGenerator(settings)
        app.state.text_generator = generator
    runner = AnalysisRunner(
        database=database,
        generator=generator,
        settings=settings,
    )
    app.state.analysis_runner = runner
    if app.state.analysis_worker_enabled:
        await runner.start()

    try:
        yield
    finally:
        await runner.stop()
        await database.dispose()
