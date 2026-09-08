from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from second_brain.db.base import Base

if TYPE_CHECKING:
    from second_brain.db.models.source import Source
    from second_brain.db.models.source_passage import SourcePassageSegment


class SourceSegment(Base):
    __tablename__ = "source_segments"
    __table_args__ = (
        CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_source_segments_page_number",
        ),
        CheckConstraint(
            "chapter_index IS NULL OR chapter_index >= 0",
            name="ck_source_segments_chapter_index",
        ),
        UniqueConstraint(
            "source_id",
            "segment_index",
            name="uq_source_segments_source_id_index",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    source_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    index: Mapped[int] = mapped_column("segment_index", Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    end_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chapter_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chapter_title: Mapped[str | None] = mapped_column(String(512), nullable=True)

    source: Mapped[Source] = relationship(back_populates="segments")
    passage_links: Mapped[list[SourcePassageSegment]] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
