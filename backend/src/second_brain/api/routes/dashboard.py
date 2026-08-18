from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import get_session
from second_brain.db.repositories.sources import dashboard_sources
from second_brain.schemas.dashboard import DashboardResponse
from second_brain.schemas.source import SourceDetail

router = APIRouter(tags=["dashboard"])


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(
    session: AsyncSession = Depends(get_session),
) -> DashboardResponse:
    source_count, recent_sources, segment_counts, knowledge_counts = await dashboard_sources(
        session
    )
    return DashboardResponse(
        source_count=source_count,
        recent_sources=[
            SourceDetail.model_validate(source).model_copy(
                update={
                    "segment_count": segment_counts.get(source.id, 0),
                    "knowledge_count": knowledge_counts.get(source.id, 0),
                }
            )
            for source in recent_sources
        ],
    )
