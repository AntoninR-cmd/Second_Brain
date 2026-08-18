from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    database: Literal["ok"] = "ok"


class OllamaStatusResponse(BaseModel):
    available: bool
    base_url: str
    configured_model: str
    model_available: bool
    error: str | None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    database: Literal["ok"] = "ok"
    ollama: OllamaStatusResponse
