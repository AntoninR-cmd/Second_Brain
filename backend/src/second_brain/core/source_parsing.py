from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from charset_normalizer import from_bytes


class SourceParseError(ValueError):
    """Raised when uploaded text cannot be decoded or parsed safely."""


@dataclass(frozen=True, slots=True)
class ParsedSubtitle:
    index: int
    text: str
    start_ms: int
    end_ms: int


_TIMESTAMP_PATTERN = r"(?P<{name}>\d{{2,}}:\d{{2}}:\d{{2}}[,.]\d{{3}})"
_TIMING_LINE = re.compile(
    rf"^\s*{_TIMESTAMP_PATTERN.format(name='start')}\s*-->\s*"
    rf"{_TIMESTAMP_PATTERN.format(name='end')}(?:[ \t]+.*)?$"
)
_BLOCK_SEPARATOR = re.compile(r"(?:\n[ \t]*){2,}")
_MAX_SQLITE_INTEGER = 2**63 - 1


def decode_text(data: bytes) -> str:
    """Decode a text payload with BOM/UTF-8 first and a conservative fallback."""

    if not data:
        raise SourceParseError("Le fichier est vide.")

    bom_encodings = (
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\xef\xbb\xbf", "utf-8-sig"),
        (b"\xfe\xff", "utf-16"),
        (b"\xff\xfe", "utf-16"),
    )
    for marker, encoding in bom_encodings:
        if data.startswith(marker):
            try:
                return _validate_decoded_text(data.decode(encoding))
            except UnicodeDecodeError as error:
                raise SourceParseError("L'encodage du fichier est invalide.") from error

    try:
        return _validate_decoded_text(data.decode("utf-8"))
    except UnicodeDecodeError:
        pass

    matches = from_bytes(data)
    match = next(
        (
            candidate
            for candidate in matches
            if candidate.encoding and candidate.encoding.lower().replace("-", "") == "cp1252"
        ),
        matches.best(),
    )
    if match is None:
        raise SourceParseError("Impossible de détecter l'encodage du fichier.")

    return _validate_decoded_text(str(match))


def _validate_decoded_text(text: str) -> str:
    if not text.strip():
        raise SourceParseError("Le fichier ne contient aucun texte.")
    if "\ufffd" in text:
        raise SourceParseError("Le fichier contient des caractères illisibles.")

    invalid_controls = [
        character
        for character in text
        if unicodedata.category(character) == "Cc" and character not in "\t\n\r"
    ]
    if invalid_controls:
        raise SourceParseError("Le fichier ne semble pas être un fichier texte valide.")
    return text


def parse_srt(data: bytes) -> tuple[str, list[ParsedSubtitle]]:
    decoded = decode_text(data)
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SourceParseError("Le fichier SRT est vide.")

    subtitles: list[ParsedSubtitle] = []
    seen_indices: set[int] = set()
    previous_index: int | None = None

    for block_number, block in enumerate(_BLOCK_SEPARATOR.split(normalized), start=1):
        lines = block.split("\n")
        if len(lines) < 3:
            raise SourceParseError(f"Bloc SRT {block_number} incomplet.")

        index_text = lines[0].strip()
        if re.fullmatch(r"[0-9]+", index_text) is None or len(index_text) > 19:
            raise SourceParseError(f"Index invalide dans le bloc SRT {block_number}.")
        index = int(index_text)
        if index < 1 or index > _MAX_SQLITE_INTEGER or index in seen_indices:
            raise SourceParseError(f"Index SRT invalide ou dupliqué : {index_text}.")
        if previous_index is not None and index <= previous_index:
            raise SourceParseError("Les index SRT doivent être strictement croissants.")

        timing = _TIMING_LINE.fullmatch(lines[1])
        if timing is None:
            raise SourceParseError(f"Timestamps invalides dans le bloc SRT {block_number}.")
        start_ms = parse_srt_timestamp(timing.group("start"))
        end_ms = parse_srt_timestamp(timing.group("end"))
        if end_ms < start_ms:
            raise SourceParseError(
                f"Le timestamp de fin précède le début dans le bloc SRT {block_number}."
            )

        subtitle_text = "\n".join(lines[2:]).strip()
        if not subtitle_text:
            raise SourceParseError(f"Texte vide dans le bloc SRT {block_number}.")

        subtitles.append(
            ParsedSubtitle(
                index=index,
                text=subtitle_text,
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )
        seen_indices.add(index)
        previous_index = index

    if not subtitles:
        raise SourceParseError("Le fichier SRT ne contient aucun sous-titre.")

    raw_text = "\n\n".join(subtitle.text for subtitle in subtitles)
    return raw_text, subtitles


def parse_srt_timestamp(value: str) -> int:
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})", value)
    if match is None:
        raise SourceParseError(f"Timestamp SRT invalide : {value}.")

    hours_text, minutes_text, seconds_text, milliseconds_text = match.groups()
    if len(hours_text) > 13:
        raise SourceParseError(f"Timestamp SRT trop grand : {value}.")
    hours, minutes, seconds, milliseconds = (
        int(hours_text),
        int(minutes_text),
        int(seconds_text),
        int(milliseconds_text),
    )
    if minutes > 59 or seconds > 59:
        raise SourceParseError(f"Timestamp SRT hors limites : {value}.")
    timestamp_ms = ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds
    if timestamp_ms > _MAX_SQLITE_INTEGER:
        raise SourceParseError(f"Timestamp SRT trop grand : {value}.")
    return timestamp_ms
