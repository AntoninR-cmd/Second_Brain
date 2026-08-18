from __future__ import annotations

import pytest
from second_brain.pipeline.chunking import (
    ChunkingConfig,
    SourceSegmentInput,
    chunk_srt_segments,
    chunk_text,
    estimate_tokens,
)


def test_short_text_produces_one_chunk_with_exact_offsets() -> None:
    text = "Une note courte, mais utile."

    chunks = chunk_text(text, ChunkingConfig(target_tokens=20, max_tokens=30))

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)
    assert chunks[0].segment_indices == ()


def test_text_chunking_prefers_sentences_and_preserves_offsets() -> None:
    text = (
        "Première phrase assez développée pour former une unité. "
        "Deuxième phrase également complète et lisible. "
        "Troisième phrase qui clôt le paragraphe.\n\n"
        "Un autre paragraphe apporte une nouvelle idée claire."
    )
    config = ChunkingConfig(target_tokens=15, max_tokens=24)

    chunks = chunk_text(text, config)

    assert len(chunks) >= 2
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    assert all(
        chunk.char_start is not None
        and chunk.char_end is not None
        and text[chunk.char_start : chunk.char_end] == chunk.text
        for chunk in chunks
    )
    assert all(chunk.text[-1] in ".!?»”" for chunk in chunks[:-1])


def test_text_chunking_uses_a_bounded_fallback_for_one_huge_token() -> None:
    text = "x" * 500
    config = ChunkingConfig(target_tokens=20, max_tokens=25)

    chunks = chunk_text(text, config)

    assert len(chunks) > 1
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text


def test_srt_long_pause_creates_boundary_and_keeps_timestamps() -> None:
    segments = [
        SourceSegmentInput(1, "Première idée complète.", start_ms=0, end_ms=1500),
        SourceSegmentInput(2, "Nouvelle section du propos.", start_ms=5000, end_ms=6500),
    ]
    config = ChunkingConfig(
        target_tokens=12,
        max_tokens=24,
        overlap_segments=0,
        pause_threshold_ms=2000,
    )

    chunks = chunk_srt_segments(segments, config)

    assert [chunk.segment_indices for chunk in chunks] == [(1,), (2,)]
    assert chunks[0].start_ms == 0
    assert chunks[0].end_ms == 1500
    assert chunks[1].start_ms == 5000
    assert chunks[1].end_ms == 6500


def test_srt_overlap_uses_whole_segments_and_advances() -> None:
    segments = [
        SourceSegmentInput(
            index,
            f"Segment numéro {index} avec une idée complète.",
            index * 1000,
            index * 1000 + 800,
        )
        for index in range(1, 7)
    ]
    config = ChunkingConfig(target_tokens=20, max_tokens=34, overlap_segments=1)

    chunks = chunk_srt_segments(segments, config)

    assert len(chunks) >= 2
    assert all(chunk.token_count <= config.max_tokens for chunk in chunks)
    assert all(chunk.segment_indices for chunk in chunks)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous.segment_indices[-1] == current.segment_indices[0]
        assert current.segment_indices[-1] > previous.segment_indices[0]


def test_srt_multiline_segment_is_not_split_when_under_limit() -> None:
    segment = SourceSegmentInput(
        7,
        "Première ligne du sous-titre.\nDeuxième ligne du même sous-titre.",
        1234,
        5678,
    )

    chunks = chunk_srt_segments(
        [segment], ChunkingConfig(target_tokens=30, max_tokens=40, overlap_segments=0)
    )

    assert len(chunks) == 1
    assert chunks[0].segment_indices == (7,)
    assert chunks[0].text == segment.text
    assert chunks[0].start_ms == 1234
    assert chunks[0].end_ms == 5678


def test_srt_segments_must_be_ordered_and_unique() -> None:
    segments = [SourceSegmentInput(2, "Deux"), SourceSegmentInput(1, "Un")]

    with pytest.raises(ValueError, match="ordonnes"):
        chunk_srt_segments(segments)


def test_token_estimate_is_empty_safe_and_monotonic() -> None:
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("une phrase") < estimate_tokens("une phrase nettement plus longue")
