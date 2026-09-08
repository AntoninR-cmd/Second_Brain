from __future__ import annotations

import posixpath
import re
import stat
import unicodedata
import zipfile
from io import BytesIO
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup, Tag
from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException

from second_brain.core.source_parsing import SourceParseError
from second_brain.parsers.base import DocumentParsingLimits, ParsedSegment, ParsedSource

_EPUB_MIMETYPE = b"application/epub+zip"
_EPUB_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_CONTAINER_PATH = "META-INF/container.xml"
_PACKAGE_XML_LIMIT = 2_000_000
_READ_CHUNK_SIZE = 64 * 1024
_SUPPORTED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_TEXT_MEDIA_TYPES = {"application/xhtml+xml", "text/html", "image/svg+xml"}
_BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"}
_HORIZONTAL_WHITESPACE = re.compile(r"[\t\f\v ]+")


def parse_epub(
    data: bytes,
    limits: DocumentParsingLimits | None = None,
) -> ParsedSource:
    """Parse an EPUB OCF container in memory and follow its package spine."""

    selected_limits = limits or DocumentParsingLimits()
    if not data.startswith(_EPUB_SIGNATURES):
        raise SourceParseError("Le fichier ne possede pas une signature EPUB/ZIP valide.")

    try:
        with zipfile.ZipFile(BytesIO(data), mode="r") as archive:
            entries = _validate_archive(archive, selected_limits)
            _validate_mimetype(archive, entries, selected_limits)
            container_xml = _read_required_member(
                archive,
                entries,
                _CONTAINER_PATH,
                min(selected_limits.epub_max_member_bytes, _PACKAGE_XML_LIMIT),
            )
            package_path = _package_path_from_container(container_xml)
            package_xml = _read_required_member(
                archive,
                entries,
                package_path,
                min(selected_limits.epub_max_member_bytes, _PACKAGE_XML_LIMIT),
            )
            package_root = _parse_xml(package_xml, "Le package OPF de l'EPUB est invalide.")
            embedded_title, embedded_author, embedded_language = _package_metadata(package_root)
            manifest = _package_manifest(package_root, package_path, entries)
            spine = _package_spine(package_root, manifest, selected_limits.epub_max_chapters)

            segments: list[ParsedSegment] = []
            extracted_characters = 0
            for item in spine:
                if "nav" in item.properties:
                    continue
                if item.media_type not in _TEXT_MEDIA_TYPES:
                    continue
                chapter_bytes = _read_required_member(
                    archive,
                    entries,
                    item.archive_path,
                    selected_limits.epub_max_member_bytes,
                )
                chapter_text, chapter_title = _extract_chapter(chapter_bytes)
                if not chapter_text:
                    continue
                extracted_characters += len(chapter_text)
                if extracted_characters > selected_limits.epub_max_extracted_chars:
                    raise SourceParseError(
                        "Le texte extrait de l'EPUB depasse la limite de securite configuree."
                    )
                chapter_index = len(segments) + 1
                segments.append(
                    ParsedSegment(
                        index=chapter_index,
                        text=chapter_text,
                        chapter_index=chapter_index,
                        chapter_title=chapter_title,
                    )
                )
    except SourceParseError:
        raise
    except (zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError) as error:
        raise SourceParseError("Le fichier EPUB est invalide ou corrompu.") from error
    except Exception as error:
        raise SourceParseError("Impossible d'extraire le texte de cet EPUB.") from error

    if not segments:
        raise SourceParseError("L'EPUB ne contient aucun chapitre textuel exploitable.")
    raw_text = "\n\n".join(segment.text for segment in segments)
    return ParsedSource(
        raw_text=raw_text,
        segments=tuple(segments),
        embedded_title=embedded_title,
        embedded_author=embedded_author,
        embedded_language=embedded_language,
        page_count=None,
        chapter_count=len(segments),
        needs_ocr=False,
    )


class _ManifestItem:
    __slots__ = ("archive_path", "media_type", "properties")

    def __init__(self, archive_path: str, media_type: str, properties: frozenset[str]) -> None:
        self.archive_path = archive_path
        self.media_type = media_type
        self.properties = properties


def _validate_archive(
    archive: zipfile.ZipFile,
    limits: DocumentParsingLimits,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos:
        raise SourceParseError("L'archive EPUB est vide.")
    if len(infos) > limits.epub_max_entries:
        raise SourceParseError("L'archive EPUB contient trop de fichiers.")

    entries: dict[str, zipfile.ZipInfo] = {}
    canonical_names: set[str] = set()
    total_uncompressed = 0
    for info in infos:
        normalized_name = _validate_archive_path(info.filename)
        canonical_name = normalized_name.casefold()
        if canonical_name in canonical_names:
            raise SourceParseError("L'archive EPUB contient des chemins dupliques ou ambigus.")
        canonical_names.add(canonical_name)

        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise SourceParseError("L'archive EPUB contient un type de fichier non autorise.")
        if info.compress_type not in _SUPPORTED_COMPRESSION:
            raise SourceParseError("L'archive EPUB utilise une compression non prise en charge.")
        if info.flag_bits & 0x1:
            raise SourceParseError("Les archives EPUB chiffrees ne sont pas prises en charge.")
        if info.file_size > limits.epub_max_member_bytes:
            raise SourceParseError("Un fichier interne de l'EPUB depasse la limite autorisee.")
        if info.file_size > 0:
            if info.compress_size <= 0:
                raise SourceParseError("L'archive EPUB presente un taux de compression dangereux.")
            if info.file_size / info.compress_size > limits.epub_max_compression_ratio:
                raise SourceParseError("L'archive EPUB presente un taux de compression dangereux.")

        total_uncompressed += info.file_size
        if total_uncompressed > limits.epub_max_uncompressed_bytes:
            raise SourceParseError("L'archive EPUB est trop volumineuse apres decompression.")
        entries[normalized_name] = info
    return entries


def _validate_archive_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise SourceParseError("L'archive EPUB contient un chemin non securise.")
    decoded = unquote(value)
    path = PurePosixPath(decoded)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SourceParseError("L'archive EPUB contient un chemin non securise.")
    if path.parts and ":" in path.parts[0]:
        raise SourceParseError("L'archive EPUB contient un chemin non securise.")
    normalized = path.as_posix()
    if normalized.startswith("../") or normalized == "..":
        raise SourceParseError("L'archive EPUB contient un chemin non securise.")
    return normalized


def _validate_mimetype(
    archive: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    limits: DocumentParsingLimits,
) -> None:
    first = archive.infolist()[0]
    mimetype = entries.get("mimetype")
    if (
        mimetype is None
        or first.filename != "mimetype"
        or mimetype.compress_type != zipfile.ZIP_STORED
    ):
        raise SourceParseError("Le conteneur EPUB ne respecte pas le format OCF attendu.")
    if _read_member(archive, mimetype, limits.epub_max_member_bytes) != _EPUB_MIMETYPE:
        raise SourceParseError("Le type de contenu interne de l'EPUB est invalide.")


def _read_required_member(
    archive: zipfile.ZipFile,
    entries: dict[str, zipfile.ZipInfo],
    path: str,
    maximum_bytes: int,
) -> bytes:
    normalized_path = _validate_archive_path(path)
    info = entries.get(normalized_path)
    if info is None or info.is_dir():
        raise SourceParseError("L'EPUB reference un fichier interne introuvable.")
    return _read_member(archive, info, maximum_bytes)


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, maximum_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    with archive.open(info, mode="r") as stream:
        while True:
            chunk = stream.read(min(_READ_CHUNK_SIZE, maximum_bytes - size + 1))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum_bytes:
                raise SourceParseError("Un fichier interne de l'EPUB depasse la limite autorisee.")
            chunks.append(chunk)
    return b"".join(chunks)


def _parse_xml(data: bytes, message: str):
    try:
        return SafeElementTree.fromstring(data)
    except (SafeElementTree.ParseError, DefusedXmlException, ValueError) as error:
        raise SourceParseError(message) from error


def _package_path_from_container(container_xml: bytes) -> str:
    root = _parse_xml(container_xml, "Le fichier container.xml de l'EPUB est invalide.")
    for element in root.iter():
        if _local_name(element.tag) != "rootfile":
            continue
        full_path = element.attrib.get("full-path", "")
        if full_path:
            return _resolve_archive_reference("", full_path)
    raise SourceParseError("L'EPUB ne reference aucun package OPF.")


def _package_metadata(root) -> tuple[str | None, str | None, str | None]:
    metadata = next((item for item in root.iter() if _local_name(item.tag) == "metadata"), None)
    if metadata is None:
        return None, None, None
    titles = _metadata_values(metadata, "title")
    creators = _metadata_values(metadata, "creator")
    languages = _metadata_values(metadata, "language")
    return (
        titles[0] if titles else None,
        "; ".join(creators)[:255] if creators else None,
        languages[0] if languages else None,
    )


def _metadata_values(metadata, local_name: str) -> list[str]:
    values: list[str] = []
    for element in metadata.iter():
        if _local_name(element.tag) != local_name or element.text is None:
            continue
        normalized = _normalize_metadata(element.text)
        if normalized:
            values.append(normalized)
    return values


def _package_manifest(
    root,
    package_path: str,
    entries: dict[str, zipfile.ZipInfo],
) -> dict[str, _ManifestItem]:
    manifest_element = next(
        (item for item in root.iter() if _local_name(item.tag) == "manifest"), None
    )
    if manifest_element is None:
        raise SourceParseError("Le package EPUB ne contient aucun manifeste.")
    package_directory = posixpath.dirname(package_path)
    manifest: dict[str, _ManifestItem] = {}
    for element in manifest_element:
        if _local_name(element.tag) != "item":
            continue
        identifier = element.attrib.get("id", "").strip()
        href = element.attrib.get("href", "").strip()
        media_type = element.attrib.get("media-type", "").strip().lower()
        if not identifier or not href or identifier in manifest:
            raise SourceParseError("Le manifeste EPUB contient un element invalide ou duplique.")
        archive_path = _resolve_archive_reference(package_directory, href)
        if archive_path not in entries:
            raise SourceParseError("Le manifeste EPUB reference un fichier interne introuvable.")
        properties = frozenset(element.attrib.get("properties", "").split())
        manifest[identifier] = _ManifestItem(archive_path, media_type, properties)
    if not manifest:
        raise SourceParseError("Le manifeste EPUB est vide.")
    return manifest


def _package_spine(
    root,
    manifest: dict[str, _ManifestItem],
    maximum_chapters: int,
) -> list[_ManifestItem]:
    spine_element = next((item for item in root.iter() if _local_name(item.tag) == "spine"), None)
    if spine_element is None:
        raise SourceParseError("Le package EPUB ne contient aucune spine.")
    spine: list[_ManifestItem] = []
    seen: set[str] = set()
    for element in spine_element:
        if _local_name(element.tag) != "itemref":
            continue
        identifier = element.attrib.get("idref", "").strip()
        if not identifier or identifier in seen or identifier not in manifest:
            raise SourceParseError("La spine EPUB contient une reference invalide ou dupliquee.")
        seen.add(identifier)
        spine.append(manifest[identifier])
        if len(spine) > maximum_chapters:
            raise SourceParseError("L'EPUB contient trop de chapitres.")
    if not spine:
        raise SourceParseError("La spine EPUB est vide.")
    return spine


def _resolve_archive_reference(base_directory: str, reference: str) -> str:
    decoded = unquote(reference)
    parsed = urlsplit(decoded)
    if parsed.scheme or parsed.netloc or parsed.query:
        raise SourceParseError("L'EPUB contient une reference externe non autorisee.")
    if "\\" in parsed.path or parsed.path.startswith("/"):
        raise SourceParseError("L'EPUB contient un chemin non securise.")
    normalized = posixpath.normpath(posixpath.join(base_directory, parsed.path))
    if not normalized or normalized in {".", ".."} or normalized.startswith("../"):
        raise SourceParseError("L'EPUB contient un chemin non securise.")
    return _validate_archive_path(normalized)


def _extract_chapter(data: bytes) -> tuple[str, str | None]:
    soup = BeautifulSoup(data, "html.parser")
    for removable in soup.find_all(("script", "style", "noscript", "template")):
        removable.decompose()
    body = soup.body or soup

    heading = body.find(("h1", "h2", "h3", "h4", "h5", "h6"))
    title = _normalize_metadata(heading.get_text(" ", strip=True)) if heading else None
    if title is None and soup.title is not None:
        title = _normalize_metadata(soup.title.get_text(" ", strip=True))

    blocks: list[str] = []
    for element in body.find_all(_BLOCK_TAGS):
        if not isinstance(element, Tag):
            continue
        if element.find_parent(_BLOCK_TAGS) is not None:
            continue
        text = _normalize_block_text(element.get_text(" ", strip=True))
        if text and (not blocks or blocks[-1] != text):
            blocks.append(text)
    if not blocks:
        fallback = _normalize_block_text(body.get_text("\n", strip=True))
        if fallback:
            blocks.append(fallback)
    return "\n\n".join(blocks), title


def _normalize_block_text(value: str) -> str:
    return _HORIZONTAL_WHITESPACE.sub(" ", value.replace("\u00a0", " ")).strip()


def _normalize_metadata(value: str) -> str | None:
    cleaned = "".join(
        " " if unicodedata.category(character) == "Cc" else character for character in value
    )
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", cleaned).strip()
    return normalized[:255] or None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
