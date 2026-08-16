from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient
from second_brain.core.config import Settings
from second_brain.main import create_app


def test_health_and_empty_dashboard(client: TestClient) -> None:
    health_response = client.get("/api/v1/system/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "database": "ok"}

    dashboard_response = client.get("/api/v1/dashboard")
    assert dashboard_response.status_code == 200
    assert dashboard_response.json() == {
        "source_count": 0,
        "recent_sources": [],
    }


def test_manual_note_is_created_listed_and_readable(client: TestClient) -> None:
    raw_text = "Première ligne\n\nContenu libre conservé."
    create_response = client.post(
        "/api/v1/sources/manual",
        json={"text": raw_text, "author": "  Ada  "},
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["type"] == "manual"
    assert created["title"] == "Première ligne"
    assert created["author"] == "Ada"
    assert created["raw_text"] == raw_text
    assert created["processing_status"] == "ready"
    created_at = created["created_at"].replace("Z", "+00:00")
    assert datetime.fromisoformat(created_at).utcoffset() is not None

    detail_response = client.get(f"/api/v1/sources/{created['id']}")
    assert detail_response.status_code == 200
    assert detail_response.json() == created

    list_response = client.get("/api/v1/sources")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["next_cursor"] is None
    assert [item["id"] for item in listed["items"]] == [created["id"]]

    dashboard_response = client.get("/api/v1/dashboard")
    assert dashboard_response.status_code == 200
    dashboard = dashboard_response.json()
    assert dashboard["source_count"] == 1
    assert dashboard["recent_sources"][0]["id"] == created["id"]
    assert dashboard["recent_sources"][0]["raw_text"] == raw_text


def test_blank_manual_note_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sources/manual",
        json={"title": "Vide", "text": " \n\t "},
    )
    assert response.status_code == 422


def test_unknown_source_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/sources/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_note_survives_a_full_application_restart(settings: Settings) -> None:
    first_app = create_app(settings)
    with TestClient(first_app) as first_client:
        response = first_client.post(
            "/api/v1/sources/manual",
            json={"title": "Persistante", "text": "Toujours présente."},
        )
        assert response.status_code == 201
        source_id = response.json()["id"]

    second_app = create_app(settings)
    with TestClient(second_app) as second_client:
        detail = second_client.get(f"/api/v1/sources/{source_id}")
        assert detail.status_code == 200
        assert detail.json()["raw_text"] == "Toujours présente."
        assert second_client.get("/api/v1/dashboard").json()["source_count"] == 1
