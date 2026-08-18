from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import get_app_settings, get_session, get_text_generator
from second_brain.core.config import Settings
from second_brain.llm.client import TextGenerator
from second_brain.schemas.system import (
    HealthResponse,
    OllamaStatusResponse,
    ReadinessResponse,
)

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(
    session: AsyncSession = Depends(get_session),
) -> HealthResponse:
    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base SQLite est indisponible.",
        ) from error
    return HealthResponse()


@router.get("/readiness", response_model=ReadinessResponse)
async def readiness(
    session: AsyncSession = Depends(get_session),
    generator: TextGenerator = Depends(get_text_generator),
    settings: Settings = Depends(get_app_settings),
) -> ReadinessResponse:
    try:
        await session.execute(text("SELECT 1"))
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="La base SQLite est indisponible.",
        ) from error

    ollama = await generator.get_readiness()
    fully_ready = ollama.ollama_available and ollama.model_available
    return ReadinessResponse(
        status="ready" if fully_ready else "degraded",
        ollama=OllamaStatusResponse(
            available=ollama.ollama_available,
            base_url=settings.ollama_base_url,
            configured_model=ollama.configured_model,
            model_available=ollama.model_available,
            error=None if fully_ready else ollama.message,
        ),
    )
