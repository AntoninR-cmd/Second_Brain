from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from second_brain.core.logging import configure_application_logging
from second_brain.db.migrations import migrate_database
from second_brain.db.session import Database
from second_brain.jobs.analysis_runner import AnalysisRunner
from second_brain.jobs.brain_runner import BrainRunner
from second_brain.jobs.indexing_runner import IndexingRunner
from second_brain.llm.client import OllamaTextGenerator
from second_brain.rag.service import RagService
from second_brain.services.brain import BrainService
from second_brain.services.vector_index import VectorIndexService
from second_brain.vector.embeddings import OllamaEmbeddingProvider
from second_brain.vector.qdrant_store import QdrantVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_application_logging()
    settings = app.state.settings
    settings.create_data_directory()

    await migrate_database(settings.resolved_database_url)
    database = Database(settings.resolved_database_url)
    app.state.database = database
    local_ai_work_lock = asyncio.Lock()
    generator = app.state.text_generator
    if generator is None:
        generator = OllamaTextGenerator(settings)
        app.state.text_generator = generator
    runner = AnalysisRunner(
        database=database,
        generator=generator,
        settings=settings,
        work_lock=local_ai_work_lock,
    )
    app.state.analysis_runner = runner
    embedding_provider = app.state.embedding_provider
    if embedding_provider is None:
        embedding_provider = OllamaEmbeddingProvider(settings)
        app.state.embedding_provider = embedding_provider
    vector_store = app.state.vector_store
    if vector_store is None:
        vector_store = QdrantVectorStore(
            settings.resolved_qdrant_path,
            reset_root=settings.resolved_data_dir,
        )
        app.state.vector_store = vector_store
    vector_service = VectorIndexService(
        database=database,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        settings=settings,
    )
    indexing_runner = IndexingRunner(
        database=database,
        service=vector_service,
        settings=settings,
        work_lock=local_ai_work_lock,
    )
    app.state.vector_index_service = vector_service
    app.state.indexing_runner = indexing_runner
    app.state.rag_service = RagService(
        database=database,
        vector_service=vector_service,
        generator=generator,
        settings=settings,
        work_lock=local_ai_work_lock,
    )
    brain_service = BrainService(
        database=database,
        vector_service=vector_service,
        generator=generator,
        settings=settings,
    )
    brain_runner = BrainRunner(
        database=database,
        service=brain_service,
        settings=settings,
        work_lock=local_ai_work_lock,
    )
    app.state.brain_service = brain_service
    app.state.brain_runner = brain_runner
    if app.state.analysis_worker_enabled:
        await runner.start()
    if app.state.indexing_worker_enabled:
        await indexing_runner.start()
    if app.state.brain_worker_enabled:
        await brain_runner.start()

    try:
        yield
    finally:
        await brain_runner.stop()
        await indexing_runner.stop()
        await runner.stop()
        await vector_store.close()
        await database.dispose()
