from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import get_session
from second_brain.db.repositories.sources import (
    create_manual_source,
    get_source,
    list_sources,
)
from second_brain.schemas.source import (
    ManualSourceCreate,
    SourceDetail,
    SourceList,
    SourceSummary,
)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post(
    "/manual",
    response_model=SourceDetail,
    status_code=status.HTTP_201_CREATED,
)
async def add_manual_source(
    payload: ManualSourceCreate,
    session: AsyncSession = Depends(get_session),
) -> SourceDetail:
    source = await create_manual_source(
        session,
        text=payload.text,
        title=payload.title,
        author=payload.author,
    )
    return SourceDetail.model_validate(source)


@router.get("", response_model=SourceList)
async def sources(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: UUID | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> SourceList:
    try:
        items, next_cursor = await list_sources(
            session,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le curseur de pagination est inconnu.",
        ) from error

    return SourceList(
        items=[SourceSummary.model_validate(item) for item in items],
        next_cursor=next_cursor,
    )


@router.get("/{source_id}", response_model=SourceDetail)
async def source_detail(
    source_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> SourceDetail:
    source = await get_source(session, source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source introuvable.",
        )
    return SourceDetail.model_validate(source)
