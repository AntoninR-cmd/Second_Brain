from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from second_brain.core.config import Settings
from second_brain.llm.client import (
    GenerationAttemptMetrics,
    GenerationCallContext,
)
from second_brain.llm.errors import OllamaInvalidResponseError
from second_brain.llm.schemas import OllamaReadiness, PassageAnalysis, SourceSummary
from second_brain.main import create_app


class CheckpointTextGenerator:
    def __init__(self, *, fail_passage_index: int | None = None) -> None:
        self.fail_passage_index = fail_passage_index
        self.passage_calls: list[int] = []

    async def get_readiness(self) -> OllamaReadiness:
        return OllamaReadiness(
            ollama_available=True,
            configured_model="qwen3.5:4b",
            model_available=True,
            available_models=["qwen3.5:4b"],
            error_code=None,
            message="Ollama et le modele sont disponibles.",
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
        del prompt, system_prompt
        passage_index = context.passage_index if context is not None else None
        if response_model is PassageAnalysis:
            assert passage_index is not None
            self.passage_calls.append(passage_index)
            if self.fail_passage_index == passage_index:
                _emit_metric(metrics_callback, call_type, outcome="invalid_response")
                raise OllamaInvalidResponseError(
                    "Validation de la reponse Ollama impossible.",
                    detail="knowledge[0].passage_indices: attendu un passage existant.",
                )
            result: Any = PassageAnalysis.model_validate(
                {
                    "passage_index": passage_index,
                    "summary": (
                        f"Resume intermediaire suffisamment detaille du passage {passage_index}."
                    ),
                    "knowledge": [
                        {
                            "title": f"Connaissance du passage {passage_index}",
                            "content": (
                                "Cette connaissance autonome est fidele au passage source "
                                f"numero {passage_index} et reste comprehensible seule."
                            ),
                            "tags": ["test"],
                            "passage_indices": [passage_index],
                        }
                    ],
                }
            )
        elif response_model is SourceSummary:
            result = SourceSummary(
                summary=(
                    "Resume final detaille et fidele construit uniquement a partir des "
                    "passages intermediaires valides de la source."
                )
            )
        else:
            raise AssertionError(f"Schema inattendu : {response_model}")

        if result_validator is not None:
            result_validator(result)
        _emit_metric(metrics_callback, call_type, outcome="success")
        return result


def _emit_metric(
    callback: Callable[[GenerationAttemptMetrics], None] | None,
    call_type: str,
    *,
    outcome: str,
) -> None:
    if callback is None:
        return
    callback(
        GenerationAttemptMetrics(
            call_type=call_type,  # type: ignore[arg-type]
            attempt=0,
            duration_seconds=0.25,
            total_duration_ns=200_000_000,
            prompt_eval_count=120,
            prompt_eval_duration_ns=40_000_000,
            eval_count=30,
            eval_duration_ns=150_000_000,
            outcome=outcome,  # type: ignore[arg-type]
        )
    )


def _settings(tmp_path: Path, name: str) -> Settings:
    return Settings(
        _env_file=None,
        env="test",
        data_dir=tmp_path,
        database_url=f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}",
        chunk_target_tokens=32,
        chunk_max_tokens=64,
        chunk_overlap_segments=0,
        job_stale_heartbeat_seconds=10,
    )


def _upload_three_passage_srt(client: TestClient) -> dict[str, Any]:
    text = (
        "Cette entree decrit une operation precise de preparation du bois avant "
        "l'application d'une finition durable et reguliere sur la surface."
    )
    payload = "\n\n".join(
        f"{index}\n00:00:{index:02d},000 --> 00:00:{index:02d},900\n{text}" for index in range(1, 4)
    )
    response = client.post(
        "/api/v1/sources/upload",
        files={"file": ("reprise.srt", payload.encode("utf-8"), "application/x-subrip")},
    )
    assert response.status_code == 201
    return response.json()


def _wait_for_source_status(
    client: TestClient,
    source_id: str,
    expected: set[str],
) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        source = client.get(f"/api/v1/sources/{source_id}").json()
        if source["analysis_status"] in expected:
            return source
        time.sleep(0.05)
    raise AssertionError("La source n'a pas atteint le statut attendu.")


def _wait_for_job_status(
    client: TestClient,
    job_id: str,
    expected: set[str],
) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in expected:
            return job
        time.sleep(0.05)
    raise AssertionError("Le traitement n'a pas atteint le statut attendu.")


def _passage_rows(database_path: Path, source_id: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            """
            SELECT passage_index, analysis_status, analysis_payload_json,
                   analysis_attempt_count, llm_call_count, knowledge_count
            FROM source_passages
            WHERE source_id = ?
            ORDER BY passage_index
            """,
            (source_id.replace("-", ""),),
        ).fetchall()


def test_restart_reuses_completed_passage_and_resumes_failed_passage(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path, "restart.sqlite3")
    database_path = tmp_path / "restart.sqlite3"
    first_generator = CheckpointTextGenerator(fail_passage_index=1)
    caplog.set_level("INFO", logger="second_brain.jobs.analysis_runner")
    caplog.set_level("INFO", logger="second_brain.services.analysis_pipeline")

    with TestClient(create_app(settings, text_generator=first_generator)) as client:
        source = _upload_three_passage_srt(client)
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze")
        assert queued.status_code == 202
        job_id = queued.json()["id"]
        failed = _wait_for_source_status(client, source["id"], {"error"})
        assert failed["knowledge_count"] == 0
        diagnostic = client.get(f"/api/v1/jobs/{job_id}").json()
        assert diagnostic["error_type"] == "OllamaInvalidResponseError"
        assert diagnostic["error_passage_index"] == 1
        assert diagnostic["error_stage"] == "passage_analysis"
        assert diagnostic["error_attempt"] == 1
        assert diagnostic["error_call_type"] == "passage_analysis"
        assert diagnostic["llm_call_count"] == 2

    failed_benchmark = next(
        record
        for record in caplog.records
        if getattr(record, "outcome", None) == "failed"
        and getattr(record, "processing_job_id", None) == job_id
    )
    assert failed_benchmark.source_passages == 3
    assert failed_benchmark.llm_calls == 2
    assert failed_benchmark.knowledge_nodes == 1
    assert failed_benchmark.prompt_eval_count == 240
    assert failed_benchmark.eval_count == 60

    rows_after_failure = _passage_rows(database_path, source["id"])
    assert [row[1] for row in rows_after_failure] == ["completed", "failed", "pending"]
    assert rows_after_failure[0][2]
    assert rows_after_failure[0][4:] == (1, 1)
    assert first_generator.passage_calls == [0, 1]

    # Simulate the same persisted job left running by an abrupt process stop. Its
    # heartbeat is deliberately fresh: process startup must still reclaim it.
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE processing_jobs
            SET status = 'running', stage = 'passage_analysis', finished_at = NULL,
                last_activity_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (job_id.replace("-", ""),),
        )
        connection.execute(
            """
            UPDATE sources
            SET analysis_status = 'processing', analysis_error = NULL
            WHERE id = ?
            """,
            (source["id"].replace("-", ""),),
        )

    resumed_generator = CheckpointTextGenerator()
    with TestClient(create_app(settings, text_generator=resumed_generator)) as client:
        analyzed = _wait_for_source_status(client, source["id"], {"analyzed", "error"})
        assert analyzed["analysis_status"] == "analyzed"
        assert analyzed["knowledge_count"] == 3
        resumed_job = _wait_for_job_status(client, job_id, {"succeeded", "failed"})
        assert resumed_job["status"] == "succeeded"
        assert resumed_job["attempt_count"] == 2
        assert resumed_job["llm_call_count"] == 5
        assert resumed_job["knowledge_node_count"] == 3

    assert resumed_generator.passage_calls == [1, 2]
    rows_after_restart = _passage_rows(database_path, source["id"])
    assert [row[1] for row in rows_after_restart] == ["completed"] * 3
    assert [row[3] for row in rows_after_restart] == [1, 2, 1]
    assert [row[4] for row in rows_after_restart] == [1, 2, 1]
    checkpoint_records = [
        record
        for record in caplog.records
        if record.message.startswith("Passage checkpoint")
        and getattr(record, "source_id", None) == source["id"]
    ]
    assert [record.passage_index for record in checkpoint_records] == [0, 1, 2]
    assert [record.knowledge_count for record in checkpoint_records] == [1, 1, 1]


def test_running_job_without_recent_heartbeat_is_reported_stale(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "stale.sqlite3")
    database_path = tmp_path / "stale.sqlite3"
    app = create_app(
        settings,
        text_generator=CheckpointTextGenerator(),
        start_analysis_worker=False,
    )
    with TestClient(app) as client:
        source = _upload_three_passage_srt(client)
        queued = client.post(f"/api/v1/sources/{source['id']}/analyze").json()
        stale_timestamp = "2020-01-01 00:00:00"
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """
                UPDATE processing_jobs
                SET status = 'running', last_activity_at = ?
                WHERE id = ?
                """,
                (stale_timestamp, queued["id"].replace("-", "")),
            )

        job = client.get(f"/api/v1/jobs/{queued['id']}").json()
        assert job["status"] == "running"
        assert job["is_stale"] is True
        assert job["heartbeat_at"] == job["last_activity_at"]
        assert datetime.fromisoformat(job["heartbeat_at"].replace("Z", "+00:00")) < datetime(
            2021,
            1,
            1,
            tzinfo=timezone.utc,
        )


def test_analysis_runner_stays_alive_after_its_queue_becomes_empty(tmp_path: Path) -> None:
    settings = _settings(tmp_path, "idle-runner.sqlite3")
    app = create_app(settings, text_generator=CheckpointTextGenerator())
    with TestClient(app) as client:
        source = _upload_three_passage_srt(client)
        assert client.post(f"/api/v1/sources/{source['id']}/analyze").status_code == 202
        analyzed = _wait_for_source_status(client, source["id"], {"analyzed", "error"})
        assert analyzed["analysis_status"] == "analyzed"

        # The loop waits at most one second for more work. On Python 3.10,
        # asyncio.TimeoutError must be caught explicitly or the runner dies here.
        time.sleep(1.2)
        runner_task = app.state.analysis_runner._task
        assert runner_task is not None
        assert runner_task.done() is False
