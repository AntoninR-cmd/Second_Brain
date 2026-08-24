from __future__ import annotations

import sqlite3
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
from second_brain.core.config import Settings
from second_brain.llm.schemas import OllamaReadiness
from second_brain.main import create_app


class ReadyTextGenerator:
    async def get_readiness(self) -> OllamaReadiness:
        return OllamaReadiness(
            ollama_available=True,
            configured_model="qwen3.5:4b",
            model_available=True,
            available_models=["qwen3.5:4b"],
            error_code=None,
            message="Ollama et le modèle sont disponibles.",
        )

    async def generate_structured(self, **kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("Le worker est désactivé dans ce test.")


def test_latest_source_analysis_exposes_persistent_progress(settings: Settings) -> None:
    app = create_app(
        settings,
        text_generator=ReadyTextGenerator(),
        start_analysis_worker=False,
    )
    with TestClient(app) as client:
        source = client.post(
            "/api/v1/sources/manual",
            json={"title": "Progression", "text": "Texte à analyser."},
        ).json()

        without_job = client.get(f"/api/v1/sources/{source['id']}/analysis")
        assert without_job.status_code == 404
        assert without_job.json()["detail"] == "Aucun traitement d'analyse pour cette source."

        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202
        latest = client.get(f"/api/v1/sources/{source['id']}/analysis")

        assert latest.status_code == 200
        assert latest.json() == queued.json()
        assert latest.json()["progress_percent"] == 0
        assert latest.json()["last_activity_at"].endswith("Z")


def test_latest_source_analysis_distinguishes_unknown_source(settings: Settings) -> None:
    app = create_app(
        settings,
        text_generator=ReadyTextGenerator(),
        start_analysis_worker=False,
    )
    with TestClient(app) as client:
        response = client.get(f"/api/v1/sources/{uuid4()}/analysis")

    assert response.status_code == 404
    assert response.json()["detail"] == "Source introuvable."


def test_latest_source_analysis_returns_the_most_recent_job(settings: Settings) -> None:
    app = create_app(
        settings,
        text_generator=ReadyTextGenerator(),
        start_analysis_worker=False,
    )
    with TestClient(app) as client:
        source = client.post(
            "/api/v1/sources/manual",
            json={"title": "Nouvelle tentative", "text": "Texte à analyser."},
        ).json()
        first = client.post(f"/api/v1/sources/{source['id']}/analyze").json()

        database_path = settings.resolved_data_dir / "second_brain.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """
                UPDATE processing_jobs
                SET status = 'failed', created_at = '2020-01-01 00:00:00'
                WHERE id = ?
                """,
                (first["id"].replace("-", ""),),
            )

        second = client.post(f"/api/v1/sources/{source['id']}/analyze").json()
        latest = client.get(f"/api/v1/sources/{source['id']}/analysis")

    assert second["id"] != first["id"]
    assert latest.status_code == 200
    assert latest.json()["id"] == second["id"]


def test_analysis_job_route_does_not_expose_vector_jobs(settings: Settings) -> None:
    app = create_app(
        settings,
        text_generator=ReadyTextGenerator(),
        start_analysis_worker=False,
    )
    with TestClient(app) as client:
        vector_job_id = uuid4().hex
        database_path = settings.resolved_data_dir / "second_brain.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """
                INSERT INTO processing_jobs (
                    id, source_id, kind, status, stage, progress_current,
                    progress_total, progress_percent, attempt_count,
                    created_at, updated_at, last_activity_at
                ) VALUES (?, NULL, 'index_knowledge', 'pending', 'queued',
                          0, 1, 0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                          CURRENT_TIMESTAMP)
                """,
                (vector_job_id,),
            )

        response = client.get(f"/api/v1/jobs/{vector_job_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Traitement introuvable."
