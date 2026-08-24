from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.core.config import Settings
from second_brain.db.session import Database
from second_brain.jobs.analysis_runner import AnalysisRunner
from second_brain.jobs.brain_runner import BrainRunner
from second_brain.jobs.indexing_runner import IndexingRunner
from second_brain.llm.client import TextGenerator
from second_brain.rag.service import RagService
from second_brain.services.brain import BrainService
from second_brain.services.vector_index import VectorIndexService
from second_brain.vector.embeddings import EmbeddingProvider
from second_brain.vector.store import VectorStore


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_text_generator(request: Request) -> TextGenerator:
    return request.app.state.text_generator


def get_analysis_runner(request: Request) -> AnalysisRunner:
    return request.app.state.analysis_runner


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_vector_index_service(request: Request) -> VectorIndexService:
    return request.app.state.vector_index_service


def get_indexing_runner(request: Request) -> IndexingRunner:
    return request.app.state.indexing_runner


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def get_brain_service(request: Request) -> BrainService:
    return request.app.state.brain_service


def get_brain_runner(request: Request) -> BrainRunner:
    return request.app.state.brain_runner


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    database = get_database(request)
    async with database.session_factory() as session:
        yield session
