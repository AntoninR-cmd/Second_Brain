from fastapi import APIRouter

from second_brain.api.routes import (
    analysis,
    brain,
    dashboard,
    knowledge,
    rag,
    search,
    sources,
    system,
    vector_index,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router)
api_router.include_router(dashboard.router)
api_router.include_router(sources.router)
api_router.include_router(analysis.router)
api_router.include_router(brain.router)
api_router.include_router(knowledge.router)
api_router.include_router(vector_index.router)
api_router.include_router(search.router)
api_router.include_router(rag.router)
