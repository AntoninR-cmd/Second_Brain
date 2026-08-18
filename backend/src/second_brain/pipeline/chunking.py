from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_END_PUNCTUATION = re.compile(r"[.!?…][\"'»”\)\]]*$")
_PARAGRAPH_SEPARATOR = re.compile(r"(?:\r?\n[ \t]*){2,}")
_SENTENCE_BOUNDARY = re.compile(r"[.!?…]+[\"'»”\)\]]*(?:\s+|$)")
_LEXICAL_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 800
    max_tokens: int = 1200
    overlap_segments: int = 2
    pause_threshold_ms: int = 2500

    def __post_init__(self) -> None:
        if self.target_tokens < 1:
            raise ValueError("target_tokens doit etre strictement positif")
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens doit etre superieur ou egal a target_tokens")
        if self.overlap_segments < 0:
            raise ValueError("overlap_segments ne peut pas etre negatif")
        if self.pause_threshold_ms < 0:
            raise ValueError("pause_threshold_ms ne peut pas etre negatif")


@dataclass(frozen=True, slots=True)
class SourceSegmentInput:
    index: int
    text: str
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("l'index de segment ne peut pas etre negatif")
        if not self.text.strip():
            raise ValueError("un segment vide ne peut pas etre decoupe")
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("start_ms ne peut pas etre negatif")
        if self.end_ms is not None and self.end_ms < 0:
            raise ValueError("end_ms ne peut pas etre negatif")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise ValueError("end_ms doit etre posterieur ou egal a start_ms")


@dataclass(frozen=True, slots=True)
class SourceChunk:
    index: int
    text: str
    token_count: int
    segment_indices: tuple[int, ...] = ()
    char_start: int | None = None
    char_end: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("l'index de passage ne peut pas etre negatif")
        if not self.text.strip():
            raise ValueError("un passage ne peut pas etre vide")
        if self.token_count < 1:
            raise ValueError("token_count doit etre strictement positif")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start et char_end doivent etre renseignes ensemble")
        if (
            self.char_start is not None
            and self.char_end is not None
            and (self.char_start < 0 or self.char_end <= self.char_start)
        ):
            raise ValueError("les offsets texte sont invalides")


@dataclass(frozen=True, slots=True)
class _TextUnit:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _SrtUnit:
    segment: SourceSegmentInput
    text: str


def estimate_tokens(text: str) -> int:
    """Return a conservative model-independent token estimate.

    Character and lexical estimates are combined so the heuristic remains useful for
    prose with long words as well as languages where spaces are less frequent.
    """

    normalized = text.strip()
    if not normalized:
        return 0
    character_estimate = len(normalized) / 4
    lexical_estimate = len(_LEXICAL_TOKEN.findall(normalized)) * 1.25
    return max(1, math.ceil(max(character_estimate, lexical_estimate)))


def chunk_text(text: str, config: ChunkingConfig | None = None) -> list[SourceChunk]:
    """Chunk TXT/manual content on paragraph, sentence, then word boundaries."""

    selected_config = config or ChunkingConfig()
    if not text.strip():
        return []

    units = _text_units(text, selected_config.max_tokens)
    chunks: list[SourceChunk] = []
    current: list[_TextUnit] = []

    def emit() -> None:
        if not current:
            return
        start = current[0].start
        end = current[-1].end
        chunk_text_value = text[start:end]
        chunks.append(
            SourceChunk(
                index=len(chunks),
                text=chunk_text_value,
                token_count=estimate_tokens(chunk_text_value),
                char_start=start,
                char_end=end,
            )
        )
        current.clear()

    for unit in units:
        if current:
            current_text = text[current[0].start : current[-1].end]
            candidate_text = text[current[0].start : unit.end]
            if (
                estimate_tokens(current_text) >= selected_config.target_tokens
                or estimate_tokens(candidate_text) > selected_config.max_tokens
            ):
                emit()
        current.append(unit)

    emit()
    return chunks


def chunk_srt_segments(
    segments: Sequence[SourceSegmentInput],
    config: ChunkingConfig | None = None,
) -> list[SourceChunk]:
    """Chunk SRT entries without losing segment indices or timestamps."""

    selected_config = config or ChunkingConfig()
    if not segments:
        return []

    _validate_segment_order(segments)
    units = _srt_units(segments, selected_config.max_tokens)
    chunks: list[SourceChunk] = []
    current: list[_SrtUnit] = []

    def current_text() -> str:
        return "\n".join(unit.text for unit in current)

    def emit() -> list[_SrtUnit]:
        if not current:
            return []
        text = current_text()
        segment_indices = tuple(dict.fromkeys(unit.segment.index for unit in current))
        starts = [unit.segment.start_ms for unit in current if unit.segment.start_ms is not None]
        ends = [unit.segment.end_ms for unit in current if unit.segment.end_ms is not None]
        chunks.append(
            SourceChunk(
                index=len(chunks),
                text=text,
                token_count=estimate_tokens(text),
                segment_indices=segment_indices,
                start_ms=min(starts) if starts else None,
                end_ms=max(ends) if ends else None,
            )
        )
        return _overlap_tail(current, selected_config.overlap_segments)

    for unit in units:
        current_is_overlap = False
        if current and _should_break_before(current, unit, selected_config):
            current = emit()
            current_is_overlap = True

        if (
            current
            and estimate_tokens(f"{current_text()}\n{unit.text}") > selected_config.max_tokens
        ):
            if not current_is_overlap:
                current = emit()
            while (
                current
                and estimate_tokens(f"{current_text()}\n{unit.text}") > selected_config.max_tokens
            ):
                current.pop(0)

        current.append(unit)

    if current:
        emit()
    return chunks


def _text_units(text: str, max_tokens: int) -> list[_TextUnit]:
    paragraph_spans = _paragraph_spans(text)
    units: list[_TextUnit] = []
    for start, end in paragraph_spans:
        paragraph = text[start:end]
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(_TextUnit(start, end))
            continue

        for sentence_start, sentence_end in _sentence_spans(text, start, end):
            sentence = text[sentence_start:sentence_end]
            if estimate_tokens(sentence) <= max_tokens:
                units.append(_TextUnit(sentence_start, sentence_end))
            else:
                units.extend(_word_spans(text, sentence_start, sentence_end, max_tokens))
    return units


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in _PARAGRAPH_SEPARATOR.finditer(text):
        span = _trim_span(text, cursor, separator.start())
        if span:
            spans.append(span)
        cursor = separator.end()
    span = _trim_span(text, cursor, len(text))
    if span:
        spans.append(span)
    return spans


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    paragraph = text[start:end]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _SENTENCE_BOUNDARY.finditer(paragraph):
        boundary_end = boundary.end()
        while boundary_end > boundary.start() and paragraph[boundary_end - 1].isspace():
            boundary_end -= 1
        span = _trim_span(text, start + cursor, start + boundary_end)
        if span:
            spans.append(span)
        cursor = boundary.end()
    span = _trim_span(text, start + cursor, end)
    if span:
        spans.append(span)
    return spans


def _word_spans(text: str, start: int, end: int, max_tokens: int) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    words = list(re.finditer(r"\S+", text[start:end]))
    current_start: int | None = None
    current_end: int | None = None

    for word in words:
        word_start = start + word.start()
        word_end = start + word.end()
        if estimate_tokens(text[word_start:word_end]) > max_tokens:
            if current_start is not None and current_end is not None:
                units.append(_TextUnit(current_start, current_end))
                current_start = current_end = None
            units.extend(_hard_limit_span(text, word_start, word_end, max_tokens))
            continue

        if current_start is None:
            current_start, current_end = word_start, word_end
        elif estimate_tokens(text[current_start:word_end]) <= max_tokens:
            current_end = word_end
        else:
            if current_end is not None:
                units.append(_TextUnit(current_start, current_end))
            current_start, current_end = word_start, word_end

    if current_start is not None and current_end is not None:
        units.append(_TextUnit(current_start, current_end))
    return units


def _hard_limit_span(text: str, start: int, end: int, max_tokens: int) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    cursor = start
    approximate_characters = max(1, max_tokens * 3)
    while cursor < end:
        candidate_end = min(end, cursor + approximate_characters)
        while (
            candidate_end > cursor + 1 and estimate_tokens(text[cursor:candidate_end]) > max_tokens
        ):
            candidate_end -= 1
        units.append(_TextUnit(cursor, candidate_end))
        cursor = candidate_end
    return units


def _srt_units(segments: Sequence[SourceSegmentInput], max_tokens: int) -> list[_SrtUnit]:
    units: list[_SrtUnit] = []
    for segment in segments:
        normalized_text = re.sub(r"[ \t]+", " ", segment.text.strip())
        if estimate_tokens(normalized_text) <= max_tokens:
            units.append(_SrtUnit(segment=segment, text=normalized_text))
            continue
        for span in _text_units(normalized_text, max_tokens):
            units.append(_SrtUnit(segment=segment, text=normalized_text[span.start : span.end]))
    return units


def _should_break_before(
    current: Sequence[_SrtUnit],
    next_unit: _SrtUnit,
    config: ChunkingConfig,
) -> bool:
    current_tokens = estimate_tokens("\n".join(unit.text for unit in current))
    previous = current[-1]
    if previous.segment.index == next_unit.segment.index:
        return False

    pause_ms: int | None = None
    if previous.segment.end_ms is not None and next_unit.segment.start_ms is not None:
        pause_ms = next_unit.segment.start_ms - previous.segment.end_ms

    if pause_ms is not None and pause_ms >= config.pause_threshold_ms:
        return current_tokens >= max(1, config.target_tokens // 3)
    if _END_PUNCTUATION.search(previous.text):
        return current_tokens >= max(1, math.ceil(config.target_tokens * 0.7))
    return current_tokens >= config.target_tokens


def _overlap_tail(units: Sequence[_SrtUnit], overlap_segments: int) -> list[_SrtUnit]:
    if overlap_segments == 0:
        return []
    selected_indices: list[int] = []
    for unit in reversed(units):
        if unit.segment.index not in selected_indices:
            selected_indices.append(unit.segment.index)
            if len(selected_indices) >= overlap_segments:
                break
    selected = set(selected_indices)
    return [unit for unit in units if unit.segment.index in selected]


def _validate_segment_order(segments: Sequence[SourceSegmentInput]) -> None:
    indices = [segment.index for segment in segments]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise ValueError("les segments SRT doivent etre ordonnes et avoir des indices uniques")


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return start, end
