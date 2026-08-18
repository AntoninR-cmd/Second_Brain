from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from second_brain.core.config import Settings
from second_brain.core.source_import import import_uploaded_source
from second_brain.main import create_app


def test_txt_import_keeps_original_bytes_and_exposes_source(
    client: TestClient,
    settings: Settings,
) -> None:
    original = "Résumé de l'été : 25 € — texte conservé.".encode("cp1252")
    response = client.post(
        "/api/v1/sources/upload",
        data={"author": "  Alice  "},
        files={"file": ("compte-rendu.txt", original, "text/plain")},
    )

    assert response.status_code == 201
    source = response.json()
    assert source["type"] == "txt"
    assert source["title"] == "compte-rendu"
    assert source["author"] == "Alice"
    assert source["original_filename"] == "compte-rendu.txt"
    assert source["raw_text"] == "Résumé de l'été : 25 € — texte conservé."
    assert source["segment_count"] == 0
    assert source["file_sha256"] == hashlib.sha256(original).hexdigest()

    stored_path = settings.resolved_data_dir / source["original_file_path"]
    assert stored_path.read_bytes() == original

    listed = client.get("/api/v1/sources").json()["items"]
    assert listed[0]["id"] == source["id"]
    assert listed[0]["type"] == "txt"
    assert listed[0]["original_filename"] == "compte-rendu.txt"


def test_srt_import_creates_segments_and_supports_bounded_pagination(
    client: TestClient,
    settings: Settings,
) -> None:
    original = (
        b"1\r\n00:00:01,250 --> 00:00:03,000\r\nBonjour\r\nle monde\r\n\r\n"
        b"2\r\n00:00:04,500 --> 00:00:06,750\r\nAu revoir\r\n"
    )
    response = client.post(
        "/api/v1/sources/upload",
        data={"title": "  Entretien  ", "author": "Ada"},
        files={"file": ("entretien.srt", original, "application/x-subrip")},
    )

    assert response.status_code == 201
    source = response.json()
    assert source["type"] == "srt"
    assert source["title"] == "Entretien"
    assert source["raw_text"] == "Bonjour\nle monde\n\nAu revoir"
    assert source["segment_count"] == 2
    assert (settings.resolved_data_dir / source["original_file_path"]).read_bytes() == original

    first_page = client.get(
        f"/api/v1/sources/{source['id']}/segments",
        params={"limit": 1},
    )
    assert first_page.status_code == 200
    first_payload = first_page.json()
    assert first_payload["next_cursor"] == 1
    assert first_payload["items"] == [
        {
            "id": first_payload["items"][0]["id"],
            "source_id": source["id"],
            "index": 1,
            "text": "Bonjour\nle monde",
            "start_ms": 1_250,
            "end_ms": 3_000,
        }
    ]

    second_page = client.get(
        f"/api/v1/sources/{source['id']}/segments",
        params={"limit": 1, "cursor": first_payload["next_cursor"]},
    ).json()
    assert second_page["next_cursor"] is None
    assert second_page["items"][0]["index"] == 2
    assert second_page["items"][0]["start_ms"] == 4_500
    assert second_page["items"][0]["end_ms"] == 6_750

    dashboard_source = client.get("/api/v1/dashboard").json()["recent_sources"][0]
    assert dashboard_source["id"] == source["id"]
    assert dashboard_source["segment_count"] == 2


def test_same_original_filename_never_causes_a_storage_collision(
    client: TestClient,
    settings: Settings,
) -> None:
    paths = []
    for text in (b"Premier contenu", b"Second contenu"):
        response = client.post(
            "/api/v1/sources/upload",
            files={"file": ("meme-nom.txt", text, "text/plain")},
        )
        assert response.status_code == 201
        paths.append(response.json()["original_file_path"])

    assert paths[0] != paths[1]
    assert (settings.resolved_data_dir / paths[0]).read_bytes() == b"Premier contenu"
    assert (settings.resolved_data_dir / paths[1]).read_bytes() == b"Second contenu"


@pytest.mark.parametrize(
    ("filename", "payload", "expected_status"),
    [
        ("document.pdf", b"not supported", 415),
        ("../escape.txt", b"text", 422),
        ("vide.txt", b"", 422),
        ("blanc.txt", b" \r\n\t ", 422),
        ("binaire.txt", b"\x00\x01\x02binary", 422),
        ("invalide.srt", b"not an SRT", 422),
    ],
)
def test_invalid_upload_is_rejected_without_creating_a_source(
    client: TestClient,
    settings: Settings,
    filename: str,
    payload: bytes,
    expected_status: int,
) -> None:
    response = client.post(
        "/api/v1/sources/upload",
        files={"file": (filename, payload, "application/octet-stream")},
    )

    assert response.status_code == expected_status
    assert client.get("/api/v1/sources").json()["items"] == []
    originals = settings.resolved_data_dir / "originals"
    assert not originals.exists() or list(originals.iterdir()) == []


def test_oversized_upload_is_rejected_with_413(
    settings: Settings,
) -> None:
    settings.max_upload_mb = 1
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/sources/upload",
            files={"file": ("large.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )

    assert response.status_code == 413


def test_srt_source_segments_and_original_survive_restart(settings: Settings) -> None:
    original = b"1\n00:00:01,000 --> 00:00:02,000\nPersistant\n"
    with TestClient(create_app(settings)) as first_client:
        created_response = first_client.post(
            "/api/v1/sources/upload",
            files={"file": ("persistant.srt", original, "application/x-subrip")},
        )
        assert created_response.status_code == 201
        created = created_response.json()

    with TestClient(create_app(settings)) as second_client:
        detail = second_client.get(f"/api/v1/sources/{created['id']}")
        segments = second_client.get(f"/api/v1/sources/{created['id']}/segments")

    assert detail.status_code == 200
    assert detail.json()["raw_text"] == "Persistant"
    assert detail.json()["segment_count"] == 1
    assert segments.status_code == 200
    assert segments.json()["items"][0]["start_ms"] == 1_000
    assert (settings.resolved_data_dir / created["original_file_path"]).read_bytes() == original


def test_segments_of_unknown_source_return_404(client: TestClient) -> None:
    response = client.get("/api/v1/sources/00000000-0000-0000-0000-000000000000/segments")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_database_failure_removes_the_copied_original(tmp_path) -> None:
    class FailingSession:
        rolled_back = False

        def add(self, source) -> None:
            del source

        async def commit(self) -> None:
            raise RuntimeError("database unavailable")

        async def rollback(self) -> None:
            self.rolled_back = True

    session = FailingSession()
    with pytest.raises(RuntimeError, match="database unavailable"):
        await import_uploaded_source(
            session,
            data_dir=tmp_path,
            filename="cleanup.txt",
            data=b"Contenu",
            title=None,
            author=None,
        )

    assert session.rolled_back is True
    originals = tmp_path / "originals"
    assert originals.is_dir()
    assert list(originals.iterdir()) == []
