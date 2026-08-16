from fastapi import APIRouter

from second_brain.api.routes import dashboard, sources, system

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(system.router)
api_router.include_router(dashboard.router)
api_router.include_router(sources.router)
