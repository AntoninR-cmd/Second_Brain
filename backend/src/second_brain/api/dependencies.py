from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.core.config import Settings
from second_brain.db.session import Database
from second_brain.jobs.analysis_runner import AnalysisRunner
from second_brain.llm.client import TextGenerator


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_text_generator(request: Request) -> TextGenerator:
    return request.app.state.text_generator


def get_analysis_runner(request: Request) -> AnalysisRunner:
    return request.app.state.analysis_runner


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    database = get_database(request)
    async with database.session_factory() as session:
        yield session
