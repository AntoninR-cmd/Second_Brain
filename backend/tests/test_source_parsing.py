from __future__ import annotations

import pytest
from second_brain.core.source_parsing import (
    SourceParseError,
    decode_text,
    parse_srt,
    parse_srt_timestamp,
)


def test_srt_parser_preserves_precise_timestamps_and_multiline_text() -> None:
    payload = (
        b"\xef\xbb\xbf1\r\n"
        b"00:00:01,250 --> 00:00:03,075\r\n"
        b"Premiere ligne\r\nDeuxieme ligne\r\n\r\n"
        b"2\r\n01:02:03,004 --> 01:02:05,999\r\nSuite\r\n"
    )

    raw_text, subtitles = parse_srt(payload)

    assert raw_text == "Premiere ligne\nDeuxieme ligne\n\nSuite"
    assert [(item.index, item.start_ms, item.end_ms) for item in subtitles] == [
        (1, 1_250, 3_075),
        (2, 3_723_004, 3_725_999),
    ]
    assert subtitles[0].text == "Premiere ligne\nDeuxieme ligne"


def test_srt_timestamp_validation() -> None:
    assert parse_srt_timestamp("00:00:00,001") == 1
    assert parse_srt_timestamp("12:34:56.789") == 45_296_789

    with pytest.raises(SourceParseError, match="hors limites"):
        parse_srt_timestamp("00:60:00,000")
    with pytest.raises(SourceParseError, match="trop grand"):
        parse_srt_timestamp("999999999999999:00:00,000")


@pytest.mark.parametrize(
    "payload",
    [
        b"1\nnot a timing line\nText",
        b"1\n00:00:02,000 --> 00:00:01,000\nText",
        (b"1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n1\n00:00:03,000 --> 00:00:04,000\nDuplicate"),
        b"1\n00:00:01,000 --> 00:00:02,000\n",
    ],
)
def test_srt_parser_rejects_every_invalid_block(payload: bytes) -> None:
    with pytest.raises(SourceParseError):
        parse_srt(payload)


def test_text_decoder_handles_utf8_bom_utf16_and_legacy_encoding() -> None:
    assert decode_text(b"\xef\xbb\xbfTexte UTF-8") == "Texte UTF-8"
    assert decode_text("Texte UTF-16".encode("utf-16")) == "Texte UTF-16"

    legacy = "Résumé de l'été : 25 € — déjà lu.".encode("cp1252")
    assert decode_text(legacy) == "Résumé de l'été : 25 € — déjà lu."


def test_text_decoder_rejects_empty_and_binary_payloads() -> None:
    with pytest.raises(SourceParseError, match="vide"):
        decode_text(b"")
    with pytest.raises(SourceParseError, match="texte"):
        decode_text(b"\x00\x01\x02binary")
