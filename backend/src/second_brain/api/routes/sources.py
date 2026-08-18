from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.api.dependencies import get_app_settings, get_session
from second_brain.core.config import Settings
from second_brain.core.source_import import (
    SourceImportError,
    import_uploaded_source,
    read_limited_upload,
    validate_filename,
)
from second_brain.db.repositories.analysis import count_source_knowledge
from second_brain.db.repositories.sources import (
    count_source_segments,
    create_manual_source,
    get_source,
    list_source_segments,
    list_sources,
)
from second_brain.schemas.source import (
    ManualSourceCreate,
    SourceDetail,
    SourceList,
    SourceSegmentList,
    SourceSegmentOut,
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


@router.post(
    "/upload",
    response_model=SourceDetail,
    status_code=status.HTTP_201_CREATED,
)
async def upload_source(
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=255),
    author: str | None = Form(default=None, max_length=255),
    settings: Settings = Depends(get_app_settings),
    session: AsyncSession = Depends(get_session),
) -> SourceDetail:
    try:
        validate_filename(file.filename)
        data = await read_limited_upload(file, settings.max_upload_bytes)
        source = await import_uploaded_source(
            session,
            data_dir=settings.resolved_data_dir,
            filename=file.filename,
            data=data,
            title=title,
            author=author,
        )
    except SourceImportError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message) from error

    segment_count = await count_source_segments(session, source.id)
    knowledge_count = await count_source_knowledge(session, source.id)
    return SourceDetail.model_validate(source).model_copy(
        update={"segment_count": segment_count, "knowledge_count": knowledge_count}
    )


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
    segment_count = await count_source_segments(session, source.id)
    knowledge_count = await count_source_knowledge(session, source.id)
    return SourceDetail.model_validate(source).model_copy(
        update={"segment_count": segment_count, "knowledge_count": knowledge_count}
    )


@router.get("/{source_id}/segments", response_model=SourceSegmentList)
async def source_segments(
    source_id: UUID,
    limit: int = Query(default=100, ge=1, le=500),
    cursor: int | None = Query(default=None, ge=0),
    session: AsyncSession = Depends(get_session),
) -> SourceSegmentList:
    source = await get_source(session, source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source introuvable.",
        )

    items, next_cursor = await list_source_segments(
        session,
        source_id=source_id,
        limit=limit,
        cursor=cursor,
    )
    return SourceSegmentList(
        items=[SourceSegmentOut.model_validate(item) for item in items],
        next_cursor=next_cursor,
    )
