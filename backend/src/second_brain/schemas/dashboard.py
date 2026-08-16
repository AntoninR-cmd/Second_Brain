from pydantic import BaseModel

from second_brain.schemas.source import SourceDetail


class DashboardResponse(BaseModel):
    source_count: int
    recent_sources: list[SourceDetail]
