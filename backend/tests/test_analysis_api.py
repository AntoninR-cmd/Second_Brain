from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from second_brain.core.config import Settings
from second_brain.llm.client import (
    GenerationAttemptMetrics,
    GenerationCallContext,
)
from second_brain.llm.errors import OllamaHTTPError, OllamaInvalidResponseError
from second_brain.llm.schemas import OllamaReadiness, PassageAnalysis, SourceSummary
from second_brain.main import create_app


class FakeTextGenerator:
    def __init__(
        self,
        *,
        available: bool = True,
        model_available: bool = True,
        mode: str = "success",
        repeated_knowledge: bool = False,
    ) -> None:
        self.available = available
        self.model_available = model_available
        self.mode = mode
        self.repeated_knowledge = repeated_knowledge
        self.passage_calls = 0
        self.call_types: list[str] = []

    async def get_readiness(self) -> OllamaReadiness:
        if not self.available:
            message = "Ollama est indisponible pour ce test."
        elif not self.model_available:
            message = "Le modèle qwen3.5:4b est absent pour ce test."
        else:
            message = "Ollama et le modèle sont disponibles."
        return OllamaReadiness(
            ollama_available=self.available,
            configured_model="qwen3.5:4b",
            model_available=self.model_available,
            available_models=["qwen3.5:4b"] if self.model_available else [],
            error_code=None if self.available else "unavailable",
            message=message,
        )

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        call_type: str,
        system_prompt: str | None = None,
        context: GenerationCallContext | None = None,
        metrics_callback: Callable[[GenerationAttemptMetrics], None] | None = None,
        result_validator: Callable[[Any], None] | None = None,
    ) -> Any:
        del context, metrics_callback
        assert system_prompt
        self.call_types.append(call_type)
        if self.mode == "error":
            raise OllamaInvalidResponseError("Réponse JSON Ollama invalide.")
        if self.mode == "unexpected_error":
            raise RuntimeError("MARQUEUR_SOURCE_STRICTEMENT_PRIVE")
        if self.mode == "http_private_error":
            raise OllamaHTTPError(
                "Ollama a repondu avec une erreur HTTP 500.",
                status_code=500,
                detail="MARQUEUR_SOURCE_STRICTEMENT_PRIVE",
            )
        if response_model is PassageAnalysis:
            passage_index = _passage_index_from_prompt(prompt)
            self.passage_calls += 1
            referenced_index = passage_index + 1 if self.mode == "bad_provenance" else passage_index
            if self.repeated_knowledge:
                title = "Préparer une finition durable"
                content = (
                    "Une préparation progressive de la surface améliore la régularité "
                    "et la durabilité de la finition appliquée ensuite."
                )
            else:
                title = f"Connaissance autonome du passage {passage_index}"
                content = (
                    f"Le passage {passage_index} expose une information autonome, fidèle "
                    "et directement justifiée par le texte conservé."
                )
            knowledge = []
            if self.mode != "empty_knowledge":
                knowledge = [
                    {
                        "title": title,
                        "content": content,
                        "tags": ["#Bois", "bois", " Finition "],
                        "passage_indices": [referenced_index],
                    }
                ]
            if self.mode == "too_many_knowledge":
                knowledge *= 3
            result = PassageAnalysis.model_validate(
                {
                    "passage_index": passage_index,
                    "summary": (
                        f"Résumé intermédiaire fidèle du passage {passage_index}, "
                        "avec les informations importantes du contenu original."
                    ),
                    "knowledge": knowledge,
                }
            )
            if result_validator is not None:
                result_validator(result)
            return result
        if response_model is SourceSummary:
            result = SourceSummary(
                summary=(
                    "Résumé final détaillé et fidèle de la source, construit uniquement "
                    "à partir des résumés intermédiaires validés."
                )
            )
            if result_validator is not None:
                result_validator(result)
            return result
        raise AssertionError(f"Schéma inattendu : {response_model}")


class BlockingFakeTextGenerator(FakeTextGenerator):
    def __init__(self, *, block_on_passage_call: int = 1) -> None:
        super().__init__()
        self.block_on_passage_call = block_on_passage_call
        self.seen_passage_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[Any],
        call_type: str,
        system_prompt: str | None = None,
        context: GenerationCallContext | None = None,
        metrics_callback: Callable[[GenerationAttemptMetrics], None] | None = None,
        result_validator: Callable[[Any], None] | None = None,
    ) -> Any:
        if response_model is PassageAnalysis:
            self.seen_passage_calls += 1
            if self.seen_passage_calls == self.block_on_passage_call:
                self.entered.set()
                released = await asyncio.to_thread(self.release.wait, 5)
                if not released:
                    raise AssertionError("Le faux Ollama n'a pas été libéré à temps.")
        return await super().generate_structured(
            prompt=prompt,
            response_model=response_model,
            call_type=call_type,
            system_prompt=system_prompt,
            context=context,
            metrics_callback=metrics_callback,
            result_validator=result_validator,
        )


def _passage_index_from_prompt(prompt: str) -> int:
    marker = "Indice du passage : "
    start = prompt.index(marker) + len(marker)
    return int(prompt[start:].splitlines()[0])


def _wait_for_terminal_source(client: TestClient, source_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/sources/{source_id}")
        assert response.status_code == 200
        source = response.json()
        if source["analysis_status"] in {"analyzed", "error"}:
            return source
        time.sleep(0.05)
    raise AssertionError("Le traitement n'a pas atteint un état terminal.")


def _manual_source(client: TestClient, title: str = "Note à analyser") -> dict[str, Any]:
    response = client.post(
        "/api/v1/sources/manual",
        json={
            "title": title,
            "text": (
                "Une préparation progressive du bois permet d'obtenir une finition "
                "plus régulière et plus durable."
            ),
        },
    )
    assert response.status_code == 201
    return response.json()


def test_application_remains_usable_when_ollama_is_stopped(settings: Settings) -> None:
    generator = FakeTextGenerator(available=False, model_available=False)
    with TestClient(create_app(settings, text_generator=generator)) as client:
        readiness = client.get("/api/v1/system/readiness")
        assert readiness.status_code == 200
        assert readiness.json() == {
            "status": "degraded",
            "database": "ok",
            "ollama": {
                "available": False,
                "base_url": "http://127.0.0.1:11434",
                "configured_model": "qwen3.5:4b",
                "model_available": False,
                "error": "Ollama est indisponible pour ce test.",
            },
        }
        source = _manual_source(client)
        rejected = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert rejected.status_code == 503
        assert client.get("/api/v1/system/health").status_code == 200
        assert (
            client.get(f"/api/v1/sources/{source['id']}").json()["analysis_status"]
            == "not_analyzed"
        )


def test_missing_configured_model_is_reported_without_downloading(settings: Settings) -> None:
    generator = FakeTextGenerator(model_available=False)
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client)
        response = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert response.status_code == 409
        assert "absent" in response.json()["detail"]
        assert generator.passage_calls == 0


def test_manual_analysis_persists_summary_nodes_tags_and_text_provenance(
    settings: Settings,
) -> None:
    generator = FakeTextGenerator()
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client)
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202
        job_id = queued.json()["id"]

        analyzed = _wait_for_terminal_source(client, source["id"])
        assert analyzed["analysis_status"] == "analyzed"
        assert analyzed["summary"].startswith("Résumé final détaillé")
        assert analyzed["knowledge_count"] == 1
        assert analyzed["raw_text"] == source["raw_text"]
        dashboard_source = client.get("/api/v1/dashboard").json()["recent_sources"][0]
        assert dashboard_source["knowledge_count"] == 1

        nodes = client.get(f"/api/v1/sources/{source['id']}/nodes").json()
        assert nodes["next_cursor"] is None
        assert len(nodes["items"]) == 1
        assert nodes["items"][0]["tags"] == ["bois", "finition"]
        assert nodes["items"][0]["evidence_count"] == 1

        node = client.get(f"/api/v1/nodes/{nodes['items'][0]['id']}").json()
        assert node["source"]["id"] == source["id"]
        assert node["source"]["original_file_path"] is None
        assert node["evidences"][0]["original_excerpt"] == source["raw_text"]
        assert node["evidences"][0]["char_start"] == 0
        assert node["evidences"][0]["char_end"] == len(source["raw_text"])

        job = client.get(f"/api/v1/jobs/{job_id}")
        assert job.status_code == 200
        assert job.json()["status"] == "succeeded"
        assert job.json()["progress_current"] == job.json()["progress_total"]
        assert generator.call_types == ["passage_analysis", "final_summary"]


def test_analysis_can_finish_with_a_summary_and_no_atomic_knowledge(
    settings: Settings,
) -> None:
    generator = FakeTextGenerator(mode="empty_knowledge")
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client, title="Note sans fait atomique")
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202

        analyzed = _wait_for_terminal_source(client, source["id"])
        assert analyzed["analysis_status"] == "analyzed"
        assert analyzed["summary"].startswith("Résumé final détaillé")
        assert analyzed["knowledge_count"] == 0
        assert client.get(f"/api/v1/sources/{source['id']}/nodes").json()["items"] == []
        job = client.get(f"/api/v1/jobs/{queued.json()['id']}").json()
        assert job["status"] == "succeeded"
        assert job["knowledge_node_count"] == 0


def test_pipeline_rejects_more_knowledge_than_the_configured_limit(
    settings: Settings,
) -> None:
    generator = FakeTextGenerator(mode="too_many_knowledge")
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client, title="Note trop prolixe")
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202

        failed = _wait_for_terminal_source(client, source["id"])
        assert failed["analysis_status"] == "error"
        job = client.get(f"/api/v1/jobs/{queued.json()['id']}").json()
        assert job["error_type"] == "StructuredOutputValidationError"
        assert "knowledge" in job["error_detail"]
        assert "au plus 2" in job["error_detail"]


def test_srt_analysis_merges_identical_knowledge_and_keeps_exact_timestamps(
    tmp_path: Path,
) -> None:
    settings = Settings(
        _env_file=None,
        env="test",
        data_dir=tmp_path,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'srt.sqlite3').as_posix()}",
        chunk_target_tokens=32,
        chunk_max_tokens=64,
        chunk_overlap_segments=0,
    )
    generator = FakeTextGenerator(repeated_knowledge=True)
    repeated = (
        "Cette entrée explique précisément une étape de préparation de la surface "
        "avant l'application soigneuse d'une finition durable."
    )
    srt = "\n\n".join(
        [
            f"1\n00:00:01,000 --> 00:00:04,250\n{repeated}",
            f"2\n00:00:05,000 --> 00:00:08,500\n{repeated}",
            f"3\n00:00:09,000 --> 00:00:12,750\n{repeated}",
        ]
    )
    with TestClient(create_app(settings, text_generator=generator)) as client:
        upload = client.post(
            "/api/v1/sources/upload",
            files={"file": ("atelier.srt", srt.encode("utf-8"), "application/x-subrip")},
        )
        assert upload.status_code == 201
        source = upload.json()
        assert client.post(f"/api/v1/sources/{source['id']}/analyze").status_code == 202

        analyzed = _wait_for_terminal_source(client, source["id"])
        assert analyzed["analysis_status"] == "analyzed"
        assert generator.passage_calls >= 2
        nodes = client.get(f"/api/v1/sources/{source['id']}/nodes").json()["items"]
        assert len(nodes) == 1
        assert nodes[0]["evidence_count"] == generator.passage_calls

        detail = client.get(f"/api/v1/nodes/{nodes[0]['id']}").json()
        assert len(detail["evidences"]) == generator.passage_calls
        assert detail["evidences"][0]["start_ms"] == 1000
        assert detail["evidences"][0]["end_ms"] == 4250
        assert detail["evidences"][0]["first_segment_index"] == 1
        assert detail["source"]["original_filename"] == "atelier.srt"
        assert detail["source"]["original_file_path"].endswith("original.srt")


def test_txt_source_analysis_keeps_file_and_character_provenance(settings: Settings) -> None:
    generator = FakeTextGenerator()
    text = (
        "Le bois doit être dépoussiéré avant l'application d'un vernis afin de conserver "
        "une surface régulière."
    )
    with TestClient(create_app(settings, text_generator=generator)) as client:
        upload = client.post(
            "/api/v1/sources/upload",
            files={"file": ("conseil.txt", text.encode("utf-8"), "text/plain")},
        )
        assert upload.status_code == 201
        source = upload.json()
        assert client.post(f"/api/v1/sources/{source['id']}/analyze").status_code == 202
        assert _wait_for_terminal_source(client, source["id"])["analysis_status"] == "analyzed"

        nodes = client.get(f"/api/v1/sources/{source['id']}/nodes").json()["items"]
        detail = client.get(f"/api/v1/nodes/{nodes[0]['id']}").json()
        assert detail["source"]["original_filename"] == "conseil.txt"
        assert detail["source"]["original_file_path"].endswith("original.txt")
        assert detail["evidences"][0]["original_excerpt"] == text
        assert detail["evidences"][0]["char_start"] == 0
        assert detail["evidences"][0]["char_end"] == len(text)


def test_invalid_structured_output_creates_no_corrupt_nodes_and_can_be_retried(
    settings: Settings,
) -> None:
    generator = FakeTextGenerator(mode="error")
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client)
        first_job = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert first_job.status_code == 202
        failed = _wait_for_terminal_source(client, source["id"])
        assert failed["analysis_status"] == "error"
        assert "JSON" in failed["analysis_error"]
        assert failed["summary"] is None
        assert failed["knowledge_count"] == 0
        assert client.get(f"/api/v1/sources/{source['id']}/nodes").json()["items"] == []
        assert client.get(f"/api/v1/jobs/{first_job.json()['id']}").json()["status"] == "failed"

        generator.mode = "success"
        retry = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert retry.status_code == 202
        retried = _wait_for_terminal_source(client, source["id"])
        assert retried["analysis_status"] == "analyzed"
        assert retried["knowledge_count"] == 1


def test_unexpected_analysis_failure_never_logs_private_exception_content(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    generator = FakeTextGenerator(mode="unexpected_error")
    caplog.set_level(logging.ERROR, logger="second_brain.jobs.analysis_runner")

    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client, "Confidentialité des journaux")
        response = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert response.status_code == 202
        failed = _wait_for_terminal_source(client, source["id"])

    assert failed["analysis_status"] == "error"
    assert failed["analysis_error"] == "Une erreur interne a interrompu l'analyse locale."
    assert "MARQUEUR_SOURCE_STRICTEMENT_PRIVE" not in caplog.text
    records = [
        record for record in caplog.records if record.name == "second_brain.jobs.analysis_runner"
    ]
    assert len(records) == 1
    assert records[0].exc_info is None
    assert "error_type=RuntimeError" in records[0].getMessage()


def test_ollama_http_error_never_persists_or_logs_private_response_detail(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    generator = FakeTextGenerator(mode="http_private_error")
    caplog.set_level(logging.ERROR, logger="second_brain.jobs.analysis_runner")

    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client, "Erreur HTTP privee")
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202
        failed = _wait_for_terminal_source(client, source["id"])
        job = client.get(f"/api/v1/jobs/{queued.json()['id']}").json()

    assert failed["analysis_error"] == "Ollama a repondu avec une erreur HTTP 500."
    assert job["error_type"] == "OllamaHTTPError"
    assert job["error_detail"] == "Ollama a repondu avec une erreur HTTP 500."
    assert "MARQUEUR_SOURCE_STRICTEMENT_PRIVE" not in caplog.text
    assert "MARQUEUR_SOURCE_STRICTEMENT_PRIVE" not in str(job)


def test_pending_analysis_is_resumed_after_application_restart(settings: Settings) -> None:
    generator = FakeTextGenerator()
    first_app = create_app(
        settings,
        text_generator=generator,
        start_analysis_worker=False,
    )
    with TestClient(first_app) as client:
        source = _manual_source(client, "Analyse persistante")
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202
        assert queued.json()["status"] == "pending"
        duplicate = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert duplicate.status_code == 202
        assert duplicate.json()["id"] == queued.json()["id"]
        assert client.get(f"/api/v1/sources/{source['id']}").json()["analysis_status"] == "queued"

    with TestClient(create_app(settings, text_generator=generator)) as client:
        analyzed = _wait_for_terminal_source(client, source["id"])
        assert analyzed["analysis_status"] == "analyzed"
        assert analyzed["knowledge_count"] == 1
        assert client.get(f"/api/v1/sources/{source['id']}/nodes").json()["items"]


def test_processing_status_is_visible_while_ollama_is_working(settings: Settings) -> None:
    generator = BlockingFakeTextGenerator()
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client)
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202
        assert generator.entered.wait(timeout=3)
        processing = client.get(f"/api/v1/sources/{source['id']}").json()
        assert processing["analysis_status"] == "processing"
        job = client.get(f"/api/v1/jobs/{queued.json()['id']}").json()
        assert job["status"] == "running"
        assert job["stage"] == "analyzing_passages"
        assert job["progress_current"] == 1
        assert job["progress_total"] == 1
        assert 0 < job["progress_percent"] < 100
        assert job["progress_message"] == "Analyse des passages : 1 / 1"
        assert job["last_activity_at"]
        generator.release.set()
        assert _wait_for_terminal_source(client, source["id"])["analysis_status"] == "analyzed"
        completed = client.get(f"/api/v1/jobs/{queued.json()['id']}").json()
        assert completed["progress_percent"] == 100


def test_persisted_progress_identifies_the_current_srt_passage(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        env="test",
        data_dir=tmp_path,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'progress.sqlite3').as_posix()}",
        chunk_target_tokens=32,
        chunk_max_tokens=64,
        chunk_overlap_segments=0,
    )
    generator = BlockingFakeTextGenerator(block_on_passage_call=2)
    passage_text = (
        "Cette entrée décrit précisément une opération de préparation du bois "
        "avant l'application d'une finition durable et régulière."
    )
    srt = "\n\n".join(
        f"{index}\n00:00:{index:02d},000 --> 00:00:{index:02d},900\n{passage_text}"
        for index in range(1, 4)
    )
    with TestClient(create_app(settings, text_generator=generator)) as client:
        upload = client.post(
            "/api/v1/sources/upload",
            files={"file": ("progression.srt", srt.encode("utf-8"), "application/x-subrip")},
        )
        assert upload.status_code == 201
        source_id = upload.json()["id"]
        assert client.post(f"/api/v1/sources/{source_id}/analyze").status_code == 202
        assert generator.entered.wait(timeout=3)

        job = client.get(f"/api/v1/sources/{source_id}/analysis").json()
        assert job["stage"] == "analyzing_passages"
        assert job["progress_current"] == 2
        assert job["progress_total"] == 3
        assert job["progress_message"] == "Analyse des passages : 2 / 3"
        assert 5 <= job["progress_percent"] < 80

        generator.release.set()
        assert _wait_for_terminal_source(client, source_id)["analysis_status"] == "analyzed"


def test_hallucinated_passage_reference_fails_traceability_validation(
    settings: Settings,
) -> None:
    generator = FakeTextGenerator(mode="bad_provenance")
    with TestClient(create_app(settings, text_generator=generator)) as client:
        source = _manual_source(client)
        assert client.post(f"/api/v1/sources/{source['id']}/analyze").status_code == 202
        failed = _wait_for_terminal_source(client, source["id"])
        assert failed["analysis_status"] == "error"
        assert "provenance" in failed["analysis_error"]
        assert failed["knowledge_count"] == 0
