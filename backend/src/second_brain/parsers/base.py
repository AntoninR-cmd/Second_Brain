from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentParsingLimits:
    """Resource limits applied while parsing untrusted document uploads."""

    pdf_max_pages: int = 10_000
    pdf_max_extracted_chars: int = 5_000_000
    pdf_min_text_characters: int = 40
    pdf_min_text_page_ratio: float = 0.10
    pdf_max_stream_bytes: int = 75_000_000
    epub_max_entries: int = 10_000
    epub_max_uncompressed_bytes: int = 200_000_000
    epub_max_member_bytes: int = 25_000_000
    epub_max_compression_ratio: float = 200.0
    epub_max_chapters: int = 5_000
    epub_max_extracted_chars: int = 5_000_000

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "pdf_max_pages",
            "pdf_max_extracted_chars",
            "pdf_max_stream_bytes",
            "epub_max_entries",
            "epub_max_uncompressed_bytes",
            "epub_max_member_bytes",
            "epub_max_chapters",
            "epub_max_extracted_chars",
        )
        for field_name in positive_integer_fields:
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} doit etre strictement positif")
        if self.pdf_min_text_characters < 0:
            raise ValueError("pdf_min_text_characters ne peut pas etre negatif")
        if not 0 <= self.pdf_min_text_page_ratio <= 1:
            raise ValueError("pdf_min_text_page_ratio doit etre compris entre 0 et 1")
        if self.epub_max_compression_ratio < 1:
            raise ValueError("epub_max_compression_ratio doit etre superieur ou egal a 1")


@dataclass(frozen=True, slots=True)
class ParsedSegment:
    """A faithful, ordered unit extracted from an original source."""

    index: int
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    page_number: int | None = None
    chapter_index: int | None = None
    chapter_title: str | None = None

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("l'index du segment doit etre strictement positif")
        if not self.text.strip():
            raise ValueError("un segment extrait ne peut pas etre vide")
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("start_ms ne peut pas etre negatif")
        if self.end_ms is not None and self.end_ms < 0:
            raise ValueError("end_ms ne peut pas etre negatif")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms doit etre posterieur ou egal a start_ms")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number doit etre strictement positif")
        if self.chapter_index is not None and self.chapter_index < 1:
            raise ValueError("chapter_index doit etre strictement positif")


@dataclass(frozen=True, slots=True)
class ParsedSource:
    """Format-neutral result of parsing a source document."""

    raw_text: str
    segments: tuple[ParsedSegment, ...]
    embedded_title: str | None = None
    embedded_author: str | None = None
    embedded_language: str | None = None
    page_count: int | None = None
    chapter_count: int | None = None
    needs_ocr: bool = False

    def __post_init__(self) -> None:
        if self.page_count is not None and self.page_count < 0:
            raise ValueError("page_count ne peut pas etre negatif")
        if self.chapter_count is not None and self.chapter_count < 0:
            raise ValueError("chapter_count ne peut pas etre negatif")
        if self.needs_ocr:
            if self.page_count is None:
                raise ValueError("needs_ocr requiert un nombre de pages")
            return
        if not self.raw_text.strip():
            raise ValueError("une source analysee ne peut pas etre vide")
        if not self.segments:
            raise ValueError("une source documentaire doit contenir des segments")
