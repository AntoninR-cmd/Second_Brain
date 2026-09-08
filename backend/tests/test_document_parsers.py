from __future__ import annotations

import zipfile
from io import BytesIO

import pytest

from second_brain.core.source_parsing import SourceParseError
from second_brain.parsers import DocumentParsingLimits, parse_epub, parse_pdf


def _pdf_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_pdf(
    pages: list[list[str]],
    *,
    title: str | None = "Livre de test",
    author: str | None = "Ada Lovelace",
) -> bytes:
    objects: list[bytes] = []
    page_object_numbers = [3 + index * 2 for index in range(len(pages))]
    font_object_number = 3 + len(pages) * 2
    info_object_number = font_object_number + 1
    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())

    for page_index, lines in enumerate(pages):
        page_object_number = page_object_numbers[page_index]
        content_object_number = page_object_number + 1
        page_object = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
            f"/Contents {content_object_number} 0 R >>"
        )
        commands = ["BT", "/F1 12 Tf", "72 720 Td"]
        for line_index, line in enumerate(lines):
            if line_index:
                commands.append("0 -20 Td")
            commands.append(f"({_pdf_literal(line)}) Tj")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        content_object = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects.extend((page_object.encode(), content_object))

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    metadata_parts = []
    if title is not None:
        metadata_parts.append(f"/Title ({_pdf_literal(title)})")
    if author is not None:
        metadata_parts.append(f"/Author ({_pdf_literal(author)})")
    objects.append(f"<< {' '.join(metadata_parts)} >>".encode("latin-1"))

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode())
        output.extend(value)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R "
            f"/Info {info_object_number} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


def _build_epub(
    *,
    chapter_order: tuple[str, ...] = ("chapter-a", "chapter-b"),
    chapter_contents: dict[str, str] | None = None,
    title: str = "EPUB de test",
    author: str = "Grace Hopper",
    language: str = "fr",
    extra_entries: tuple[tuple[zipfile.ZipInfo | str, bytes], ...] = (),
    manifest_href_override: dict[str, str] | None = None,
    container_xml: bytes | None = None,
) -> bytes:
    contents = chapter_contents or {
        "chapter-a": "<html><head><title>A</title></head><body><h1>Premier chapitre</h1><p>Le premier contenu suit l'ordre de lecture.</p></body></html>",
        "chapter-b": "<html><head><title>B</title></head><body><h1>Second chapitre</h1><p>Le second contenu vient ensuite.</p></body></html>",
    }
    href_overrides = manifest_href_override or {}
    manifest_items = "".join(
        f'<item id="{identifier}" href="{href_overrides.get(identifier, identifier + ".xhtml")}" media-type="application/xhtml+xml"/>'
        for identifier in contents
    )
    spine_items = "".join(f'<itemref idref="{identifier}"/>' for identifier in chapter_order)
    package = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>{title}</dc:title><dc:creator>{author}</dc:creator><dc:language>{language}</dc:language>
  </metadata>
  <manifest>{manifest_items}</manifest>
  <spine>{spine_items}</spine>
</package>""".encode()
    container = (
        container_xml
        or b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container, compress_type=zipfile.ZIP_DEFLATED)
        archive.writestr("OEBPS/content.opf", package, compress_type=zipfile.ZIP_DEFLATED)
        # Deliberately reverse ZIP order: the parser must use the spine, not archive order.
        for identifier, content in reversed(tuple(contents.items())):
            archive.writestr(
                f"OEBPS/{identifier}.xhtml", content.encode(), compress_type=zipfile.ZIP_DEFLATED
            )
        for name_or_info, payload in extra_entries:
            archive.writestr(name_or_info, payload)
    return buffer.getvalue()


def test_pdf_extracts_multiple_pages_metadata_and_page_provenance() -> None:
    parsed = parse_pdf(
        _build_pdf(
            [
                ["Premiere page", "Un paragraphe suffisamment long pour etre exploitable."],
                ["Deuxieme page", "Un second paragraphe conserve sa provenance exacte."],
            ]
        ),
        DocumentParsingLimits(pdf_min_text_characters=1),
    )

    assert parsed.page_count == 2
    assert parsed.chapter_count is None
    assert parsed.embedded_title == "Livre de test"
    assert parsed.embedded_author == "Ada Lovelace"
    assert parsed.needs_ocr is False
    assert [segment.page_number for segment in parsed.segments] == [1, 2]
    assert "Premiere page" in parsed.segments[0].text
    assert parsed.raw_text.index("Premiere page") < parsed.raw_text.index("Deuxieme page")


def test_pdf_removes_only_repeated_margin_lines_from_three_or_more_pages() -> None:
    pages = [
        [
            "Manuel Second Brain",
            f"Contenu propre de la page {index} avec une phrase.",
            "Confidentiel",
        ]
        for index in range(1, 4)
    ]
    parsed = parse_pdf(_build_pdf(pages), DocumentParsingLimits(pdf_min_text_characters=1))

    assert "Manuel Second Brain" not in parsed.raw_text
    assert "Confidentiel" not in parsed.raw_text
    assert all("Contenu propre" in segment.text for segment in parsed.segments)


def test_pdf_without_a_text_layer_is_marked_as_requiring_ocr() -> None:
    parsed = parse_pdf(_build_pdf([[], []]))

    assert parsed.needs_ocr is True
    assert parsed.page_count == 2
    assert parsed.raw_text == ""
    assert parsed.segments == ()


def test_zero_page_pdf_is_rejected() -> None:
    with pytest.raises(SourceParseError, match="aucune page"):
        parse_pdf(_build_pdf([]))


@pytest.mark.parametrize("payload", [b"pas un PDF", b"%PDF-1.4\ncorrompu"])
def test_invalid_or_corrupt_pdf_is_rejected(payload: bytes) -> None:
    with pytest.raises(SourceParseError, match="PDF"):
        parse_pdf(payload)


def test_pdf_enforces_page_and_extracted_text_limits() -> None:
    payload = _build_pdf([["A" * 80], ["B" * 80]])
    with pytest.raises(SourceParseError, match="trop de pages"):
        parse_pdf(payload, DocumentParsingLimits(pdf_max_pages=1))
    with pytest.raises(SourceParseError, match="texte extrait"):
        parse_pdf(payload, DocumentParsingLimits(pdf_max_extracted_chars=50))


def test_epub_preserves_spine_order_metadata_and_chapter_provenance() -> None:
    parsed = parse_epub(_build_epub(chapter_order=("chapter-b", "chapter-a")))

    assert parsed.embedded_title == "EPUB de test"
    assert parsed.embedded_author == "Grace Hopper"
    assert parsed.embedded_language == "fr"
    assert parsed.chapter_count == 2
    assert [segment.chapter_index for segment in parsed.segments] == [1, 2]
    assert [segment.chapter_title for segment in parsed.segments] == [
        "Second chapitre",
        "Premier chapitre",
    ]
    assert parsed.raw_text.index("Second contenu") < parsed.raw_text.index("premier contenu")


def test_epub_strips_executable_markup_and_keeps_paragraph_structure() -> None:
    parsed = parse_epub(
        _build_epub(
            chapter_order=("chapter-a",),
            chapter_contents={
                "chapter-a": "<html><body><h1>Chapitre</h1><script>danger()</script><style>.x{}</style><p>Premier paragraphe.</p><p>Second paragraphe.</p></body></html>"
            },
        )
    )

    assert "danger" not in parsed.raw_text
    assert ".x" not in parsed.raw_text
    assert "Premier paragraphe.\n\nSecond paragraphe." in parsed.raw_text


@pytest.mark.parametrize(
    "payload",
    [
        b"pas un EPUB",
        b"PK\x03\x04corrompu",
    ],
)
def test_invalid_or_corrupt_epub_is_rejected(payload: bytes) -> None:
    with pytest.raises(SourceParseError, match="EPUB"):
        parse_epub(payload)


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/absolu.txt", "C:/evil.txt"])
def test_epub_rejects_archive_path_traversal(unsafe_name: str) -> None:
    with pytest.raises(SourceParseError, match="chemin non securise"):
        parse_epub(_build_epub(extra_entries=((unsafe_name, b"danger"),)))


def test_epub_rejects_encoded_manifest_path_traversal() -> None:
    with pytest.raises(SourceParseError, match="chemin non securise"):
        parse_epub(
            _build_epub(
                chapter_order=("chapter-a",),
                chapter_contents={"chapter-a": "<html><body><p>Texte</p></body></html>"},
                manifest_href_override={"chapter-a": "%2e%2e/%2e%2e/secret.xhtml"},
            )
        )


def test_epub_rejects_symlinks_and_dangerous_compression_ratio() -> None:
    symlink = zipfile.ZipInfo("OEBPS/link")
    symlink.create_system = 3
    symlink.external_attr = (stat_mode := 0o120777) << 16
    assert stat_mode
    with pytest.raises(SourceParseError, match="type de fichier"):
        parse_epub(_build_epub(extra_entries=((symlink, b"target"),)))

    with pytest.raises(SourceParseError, match="taux de compression"):
        parse_epub(
            _build_epub(extra_entries=(("OEBPS/bomb.txt", b"0" * 20_000),)),
            DocumentParsingLimits(epub_max_compression_ratio=2),
        )


def test_epub_enforces_entry_chapter_and_extracted_text_limits() -> None:
    payload = _build_epub()
    with pytest.raises(SourceParseError, match="trop de fichiers"):
        parse_epub(payload, DocumentParsingLimits(epub_max_entries=2))
    with pytest.raises(SourceParseError, match="trop de chapitres"):
        parse_epub(payload, DocumentParsingLimits(epub_max_chapters=1))
    with pytest.raises(SourceParseError, match="texte extrait"):
        parse_epub(payload, DocumentParsingLimits(epub_max_extracted_chars=20))


def test_epub_rejects_empty_content_and_xml_entities() -> None:
    with pytest.raises(SourceParseError, match="aucun chapitre"):
        parse_epub(
            _build_epub(
                chapter_order=("chapter-a",),
                chapter_contents={"chapter-a": "<html><body><script>vide</script></body></html>"},
            )
        )

    entity_container = b"""<?xml version="1.0"?>
<!DOCTYPE container [<!ENTITY x "boom">]>
<container><rootfiles><rootfile full-path="&x;"/></rootfiles></container>"""
    with pytest.raises(SourceParseError, match="container.xml"):
        parse_epub(_build_epub(container_xml=entity_container))
