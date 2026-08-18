from __future__ import annotations

from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.db.models.knowledge import KnowledgeNode
from second_brain.db.models.source import ProcessingStatus, Source, SourceType
from second_brain.db.models.source_segment import SourceSegment


def derive_title(text: str) -> str:
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        "Note sans titre",
    )
    if len(first_line) <= 120:
        return first_line
    return f"{first_line[:119].rstrip()}…"


async def create_manual_source(
    session: AsyncSession,
    *,
    text: str,
    title: str | None,
    author: str | None,
) -> Source:
    source = Source(
        type=SourceType.MANUAL,
        title=title or derive_title(text),
        author=author,
        raw_text=text,
        processing_status=ProcessingStatus.READY,
    )
    session.add(source)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    await session.refresh(source)
    return source


async def get_source(session: AsyncSession, source_id: UUID) -> Source | None:
    return await session.get(Source, source_id)


async def count_source_segments(session: AsyncSession, source_id: UUID) -> int:
    count = await session.scalar(
        select(func.count(SourceSegment.id)).where(SourceSegment.source_id == source_id)
    )
    return int(count or 0)


async def list_source_segments(
    session: AsyncSession,
    *,
    source_id: UUID,
    limit: int,
    cursor: int | None,
) -> tuple[list[SourceSegment], int | None]:
    statement = select(SourceSegment).where(SourceSegment.source_id == source_id)
    if cursor is not None:
        statement = statement.where(SourceSegment.index > cursor)
    statement = statement.order_by(SourceSegment.index.asc())

    rows = list((await session.scalars(statement.limit(limit + 1))).all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].index if has_more and items else None
    return items, next_cursor


async def list_sources(
    session: AsyncSession,
    *,
    limit: int,
    cursor: UUID | None,
) -> tuple[list[Source], UUID | None]:
    statement = select(Source)

    if cursor is not None:
        boundary = await session.get(Source, cursor)
        if boundary is None:
            raise ValueError("Unknown source cursor")
        statement = statement.where(
            or_(
                Source.created_at < boundary.created_at,
                and_(
                    Source.created_at == boundary.created_at,
                    Source.id < boundary.id,
                ),
            )
        )

    statement = statement.order_by(Source.created_at.desc(), Source.id.desc())
    rows = list((await session.scalars(statement.limit(limit + 1))).all())
    has_more = len(rows) > limit
    items = rows[:limit]
    next_cursor = items[-1].id if has_more and items else None
    return items, next_cursor


async def dashboard_sources(
    session: AsyncSession,
    *,
    recent_limit: int = 5,
) -> tuple[int, list[Source], dict[UUID, int], dict[UUID, int]]:
    source_count = await session.scalar(select(func.count(Source.id)))
    recent = list(
        (
            await session.scalars(
                select(Source)
                .order_by(Source.created_at.desc(), Source.id.desc())
                .limit(recent_limit)
            )
        ).all()
    )
    segment_counts: dict[UUID, int] = {}
    knowledge_counts: dict[UUID, int] = {}
    if recent:
        source_ids = [source.id for source in recent]
        count_rows = await session.execute(
            select(SourceSegment.source_id, func.count(SourceSegment.id))
            .where(SourceSegment.source_id.in_(source_ids))
            .group_by(SourceSegment.source_id)
        )
        segment_counts = {source_id: int(count) for source_id, count in count_rows.all()}
        knowledge_rows = await session.execute(
            select(KnowledgeNode.source_id, func.count(KnowledgeNode.id))
            .where(KnowledgeNode.source_id.in_(source_ids))
            .group_by(KnowledgeNode.source_id)
        )
        knowledge_counts = {source_id: int(count) for source_id, count in knowledge_rows.all()}
    return int(source_count or 0), recent, segment_counts, knowledge_counts
