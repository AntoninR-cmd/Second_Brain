from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from second_brain.api.router import api_router
from second_brain.core.config import Settings, get_settings
from second_brain.core.lifecycle import lifespan
from second_brain.llm.client import TextGenerator


def create_app(
    settings: Settings | None = None,
    *,
    text_generator: TextGenerator | None = None,
    start_analysis_worker: bool = True,
) -> FastAPI:
    selected_settings = settings or get_settings()
    application = FastAPI(
        title=selected_settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = selected_settings
    application.state.text_generator = text_generator
    application.state.analysis_worker_enabled = start_analysis_worker

    origins = selected_settings.allowed_origin_list
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    application.include_router(api_router)
    return application


app = create_app()
