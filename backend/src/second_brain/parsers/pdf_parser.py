from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from io import BytesIO

from pypdf import PdfReader, apply_configuration
from pypdf.errors import LimitReachedError, PdfReadError

from second_brain.core.source_parsing import SourceParseError
from second_brain.parsers.base import DocumentParsingLimits, ParsedSegment, ParsedSource

_PDF_SIGNATURE = b"%PDF-"
_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")
_MULTIPLE_BLANK_LINES = re.compile(r"\n(?:[ \t]*\n){2,}")
_MARGIN_OCCURRENCE_RATIO = 0.60
_MAX_MARGIN_LINE_CHARACTERS = 160
_MIN_MEANINGFUL_CHARACTERS_PER_PAGE = 10


def parse_pdf(
    data: bytes,
    limits: DocumentParsingLimits | None = None,
) -> ParsedSource:
    """Extract a text PDF page-by-page while retaining exact page provenance."""

    selected_limits = limits or DocumentParsingLimits()
    if not data.startswith(_PDF_SIGNATURE):
        raise SourceParseError("Le fichier ne possede pas une signature PDF valide.")

    try:
        with apply_configuration(
            maximum_declared_stream_length=selected_limits.pdf_max_stream_bytes,
            array_based_stream_maximum_output_length=selected_limits.pdf_max_stream_bytes,
            jbig2_maximum_output_length=selected_limits.pdf_max_stream_bytes,
            lzw_maximum_output_length=selected_limits.pdf_max_stream_bytes,
            run_length_maximum_output_length=selected_limits.pdf_max_stream_bytes,
            zlib_maximum_output_length=selected_limits.pdf_max_stream_bytes,
            image_maximum_buffer_size=selected_limits.pdf_max_stream_bytes,
            jbig2dec_binary=None,
        ):
            reader = PdfReader(
                BytesIO(data),
                strict=False,
                root_object_recovery_limit=10_000,
            )
            try:
                if reader.is_encrypted:
                    raise SourceParseError(
                        "Les PDF proteges par mot de passe ne sont pas pris en charge."
                    )

                page_count = len(reader.pages)
                if page_count == 0:
                    raise SourceParseError("Le PDF ne contient aucune page.")
                if page_count > selected_limits.pdf_max_pages:
                    raise SourceParseError(
                        "Le PDF contient trop de pages pour etre importe en toute securite."
                    )

                page_lines: list[list[str]] = []
                extracted_characters = 0
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    normalized_lines = _normalize_page_lines(extracted)
                    extracted_characters += sum(len(line) for line in normalized_lines)
                    if extracted_characters > selected_limits.pdf_max_extracted_chars:
                        raise SourceParseError(
                            "Le texte extrait du PDF depasse la limite de securite configuree."
                        )
                    page_lines.append(normalized_lines)

                metadata = _read_metadata(reader)
            finally:
                reader.close()
    except SourceParseError:
        raise
    except (
        LimitReachedError,
        PdfReadError,
        RecursionError,
        ValueError,
        TypeError,
        OverflowError,
    ) as error:
        raise SourceParseError("Le fichier PDF est invalide ou corrompu.") from error
    except Exception as error:
        # pypdf can surface several low-level exceptions for malformed object graphs.
        # None of those implementation details should be exposed to the client.
        raise SourceParseError("Impossible d'extraire le texte de ce PDF.") from error

    cleaned_pages = _remove_repeated_margin_lines(page_lines)
    page_texts = [_lines_to_text(lines) for lines in cleaned_pages]
    meaningful_counts = [_meaningful_character_count(text) for text in page_texts]
    total_meaningful = sum(meaningful_counts)
    text_page_count = sum(
        count >= _MIN_MEANINGFUL_CHARACTERS_PER_PAGE for count in meaningful_counts
    )
    text_page_ratio = text_page_count / page_count
    needs_ocr = (
        total_meaningful < selected_limits.pdf_min_text_characters
        or text_page_ratio < selected_limits.pdf_min_text_page_ratio
    )

    if needs_ocr:
        return ParsedSource(
            raw_text="",
            segments=(),
            embedded_title=metadata[0],
            embedded_author=metadata[1],
            page_count=page_count,
            chapter_count=None,
            needs_ocr=True,
        )

    segments = tuple(
        ParsedSegment(index=page_number, text=text, page_number=page_number)
        for page_number, text in enumerate(page_texts, start=1)
        if text
    )
    raw_text = "\n\n".join(segment.text for segment in segments)
    if len(raw_text) > selected_limits.pdf_max_extracted_chars:
        raise SourceParseError("Le texte extrait du PDF depasse la limite de securite configuree.")

    return ParsedSource(
        raw_text=raw_text,
        segments=segments,
        embedded_title=metadata[0],
        embedded_author=metadata[1],
        page_count=page_count,
        chapter_count=None,
        needs_ocr=False,
    )


def _normalize_page_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines: list[str] = []
    previous_was_blank = True
    for raw_line in normalized.split("\n"):
        safe_line = "".join(
            " " if unicodedata.category(character) == "Cc" else character for character in raw_line
        )
        line = _HORIZONTAL_WHITESPACE.sub(" ", safe_line).strip()
        if line:
            lines.append(line)
            previous_was_blank = False
        elif not previous_was_blank:
            lines.append("")
            previous_was_blank = True
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _remove_repeated_margin_lines(pages: list[list[str]]) -> list[list[str]]:
    if len(pages) < 3:
        return [list(page) for page in pages]

    eligible_pages = [sum(bool(line) for line in page) >= 3 for page in pages]
    first_lines = [
        _first_non_empty(page) if eligible else None
        for page, eligible in zip(pages, eligible_pages, strict=True)
    ]
    last_lines = [
        _last_non_empty(page) if eligible else None
        for page, eligible in zip(pages, eligible_pages, strict=True)
    ]
    minimum_occurrences = max(3, math.ceil(len(pages) * _MARGIN_OCCURRENCE_RATIO))
    repeated_headers = _repeated_margin_values(first_lines, minimum_occurrences)
    repeated_footers = _repeated_margin_values(last_lines, minimum_occurrences)

    cleaned: list[list[str]] = []
    for page in pages:
        page_copy = list(page)
        first_index = _first_non_empty_index(page_copy)
        if (
            first_index is not None
            and _canonical_margin(page_copy[first_index]) in repeated_headers
        ):
            page_copy.pop(first_index)
        last_index = _last_non_empty_index(page_copy)
        if last_index is not None and _canonical_margin(page_copy[last_index]) in repeated_footers:
            page_copy.pop(last_index)
        cleaned.append(_trim_blank_lines(page_copy))
    return cleaned


def _repeated_margin_values(values: list[str | None], minimum: int) -> set[str]:
    candidates = [
        _canonical_margin(value)
        for value in values
        if value is not None and 1 < len(value) <= _MAX_MARGIN_LINE_CHARACTERS
    ]
    return {value for value, count in Counter(candidates).items() if count >= minimum}


def _canonical_margin(value: str) -> str:
    return _HORIZONTAL_WHITESPACE.sub(" ", value).strip().casefold()


def _first_non_empty(lines: list[str]) -> str | None:
    index = _first_non_empty_index(lines)
    return lines[index] if index is not None else None


def _last_non_empty(lines: list[str]) -> str | None:
    index = _last_non_empty_index(lines)
    return lines[index] if index is not None else None


def _first_non_empty_index(lines: list[str]) -> int | None:
    return next((index for index, line in enumerate(lines) if line), None)


def _last_non_empty_index(lines: list[str]) -> int | None:
    return next((index for index in range(len(lines) - 1, -1, -1) if lines[index]), None)


def _trim_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and not lines[start]:
        start += 1
    while end > start and not lines[end - 1]:
        end -= 1
    return lines[start:end]


def _lines_to_text(lines: list[str]) -> str:
    return _MULTIPLE_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _meaningful_character_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _read_metadata(reader: PdfReader) -> tuple[str | None, str | None]:
    try:
        metadata = reader.metadata
        if metadata is None:
            return None, None
        return (
            _normalize_metadata_value(getattr(metadata, "title", None)),
            _normalize_metadata_value(getattr(metadata, "author", None)),
        )
    except Exception:
        # Broken optional metadata must not make an otherwise readable PDF unusable.
        return None, None


def _normalize_metadata_value(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(
        "" if unicodedata.category(character) == "Cc" else character for character in str(value)
    ).strip()
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", normalized)
    return normalized[:255] or None
