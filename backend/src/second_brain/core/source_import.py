from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import unicodedata
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from second_brain.core.source_parsing import (
    ParsedSubtitle,
    SourceParseError,
    decode_text,
    parse_srt,
)
from second_brain.db.models.source import ProcessingStatus, Source, SourceType
from second_brain.db.models.source_segment import SourceSegment

_ALLOWED_EXTENSIONS = {".srt": SourceType.SRT, ".txt": SourceType.TXT}
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "clock$",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_INVALID_FILENAME_CHARACTERS = re.compile(r'[<>:"/\\|?*]')
_READ_CHUNK_SIZE = 1024 * 1024


class SourceImportError(Exception):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def read_limited_upload(upload: UploadFile, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(_READ_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise SourceImportError(
                "Le fichier dépasse la taille maximale autorisée "
                f"({max_bytes // 1024 // 1024} Mo).",
                status_code=413,
            )
        chunks.append(chunk)

    if not chunks:
        raise SourceImportError("Le fichier est vide.")
    return b"".join(chunks)


async def import_uploaded_source(
    session: AsyncSession,
    *,
    data_dir: Path,
    filename: str | None,
    data: bytes,
    title: str | None,
    author: str | None,
) -> Source:
    safe_filename, extension, source_type = validate_filename(filename)
    normalized_title = _normalize_metadata(title, "titre")
    normalized_author = _normalize_metadata(author, "auteur")

    try:
        raw_text, subtitles = await asyncio.to_thread(_extract_source, data, source_type)
    except SourceParseError as error:
        raise SourceImportError(str(error)) from error

    source_id = uuid4()
    relative_path = Path("originals") / str(source_id) / f"original{extension}"
    source = Source(
        id=source_id,
        type=source_type,
        title=normalized_title or _title_from_filename(safe_filename),
        author=normalized_author,
        original_filename=safe_filename,
        original_file_path=relative_path.as_posix(),
        file_sha256=hashlib.sha256(data).hexdigest(),
        raw_text=raw_text,
        processing_status=ProcessingStatus.READY,
    )
    source.segments = [
        SourceSegment(
            index=subtitle.index,
            text=subtitle.text,
            start_ms=subtitle.start_ms,
            end_ms=subtitle.end_ms,
        )
        for subtitle in subtitles
    ]

    source_directory = await asyncio.to_thread(
        _write_original_atomically,
        data_dir,
        relative_path,
        data,
    )
    try:
        session.add(source)
        await session.commit()
    except Exception:
        await session.rollback()
        await asyncio.to_thread(_remove_source_directory, data_dir, source_directory)
        raise

    await session.refresh(source)
    return source


def validate_filename(filename: str | None) -> tuple[str, str, SourceType]:
    candidate = filename or ""
    if not candidate or candidate != candidate.strip() or len(candidate) > 255:
        raise SourceImportError("Le nom du fichier est absent ou trop long.")
    if any(unicodedata.category(character) == "Cc" for character in candidate):
        raise SourceImportError("Le nom du fichier contient des caractères de contrôle.")
    if _INVALID_FILENAME_CHARACTERS.search(candidate):
        raise SourceImportError("Le nom du fichier contient un caractère interdit.")
    if candidate in {".", ".."} or candidate.endswith((".", " ")):
        raise SourceImportError("Le nom du fichier est invalide.")

    path = Path(candidate)
    extension = path.suffix.lower()
    source_type = _ALLOWED_EXTENSIONS.get(extension)
    if source_type is None:
        raise SourceImportError(
            "Seuls les fichiers .srt et .txt sont acceptés.",
            status_code=415,
        )
    if path.stem.lower() in _WINDOWS_RESERVED_NAMES:
        raise SourceImportError("Le nom du fichier est réservé par Windows.")
    return candidate, extension, source_type


def _extract_source(
    data: bytes,
    source_type: SourceType,
) -> tuple[str, list[ParsedSubtitle]]:
    if source_type is SourceType.SRT:
        return parse_srt(data)
    return decode_text(data), []


def _normalize_metadata(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if len(normalized) > 255:
        raise SourceImportError(f"Le {field_name} dépasse 255 caractères.")
    return normalized


def _title_from_filename(filename: str) -> str:
    title = Path(filename).stem.strip(" .")
    return (title or "Source importée")[:255]


def _write_original_atomically(data_dir: Path, relative_path: Path, data: bytes) -> Path:
    root = data_dir.resolve()
    originals_root = (root / "originals").resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(originals_root):
        raise RuntimeError("Le chemin de stockage calculé sort du dossier des originaux.")

    source_directory = target.parent
    source_directory.mkdir(parents=True, exist_ok=False)
    temporary_target = source_directory / ".original.tmp"
    try:
        with temporary_target.open("xb") as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_target, target)
    except Exception:
        _remove_source_directory(root, source_directory)
        raise
    return source_directory


def _remove_source_directory(data_dir: Path, source_directory: Path) -> None:
    originals_root = (data_dir.resolve() / "originals").resolve()
    resolved_directory = source_directory.resolve()
    if resolved_directory.parent != originals_root:
        raise RuntimeError("Refus de supprimer un dossier hors du stockage des originaux.")
    if resolved_directory.exists():
        shutil.rmtree(resolved_directory)
