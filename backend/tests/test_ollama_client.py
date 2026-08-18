from __future__ import annotations

import json
import logging
from uuid import UUID

import httpx2
import pytest
from second_brain.core.config import Settings
from second_brain.llm import (
    GenerationAttemptMetrics,
    GenerationCallContext,
    OllamaHTTPError,
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTextGenerator,
    OllamaTimeoutError,
    OllamaUnavailableError,
    PassageAnalysis,
    StructuredOutputValidationError,
)


def _settings(**updates: object) -> Settings:
    return Settings(_env_file=None).model_copy(update=updates)


@pytest.mark.anyio
async def test_readiness_reports_configured_model() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/tags"
        return httpx2.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3.5:4b"},
                    {"model": "autre-modele:latest"},
                ]
            },
        )

    generator = OllamaTextGenerator(_settings(), transport=httpx2.MockTransport(handler))

    readiness = await generator.get_readiness()

    assert readiness.ollama_available is True
    assert readiness.model_available is True
    assert readiness.configured_model == "qwen3.5:4b"
    assert readiness.available_models == ["qwen3.5:4b", "autre-modele:latest"]


@pytest.mark.anyio
async def test_readiness_distinguishes_missing_model_from_unavailable_ollama() -> None:
    async def missing_model_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"models": [{"name": "autre:latest"}]})

    missing_model = OllamaTextGenerator(
        _settings(), transport=httpx2.MockTransport(missing_model_handler)
    )
    readiness = await missing_model.get_readiness()

    assert readiness.ollama_available is True
    assert readiness.model_available is False
    assert "ollama pull qwen3.5:4b" in readiness.message

    async def unavailable_handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    unavailable = OllamaTextGenerator(
        _settings(), transport=httpx2.MockTransport(unavailable_handler)
    )
    readiness = await unavailable.get_readiness()

    assert readiness.ollama_available is False
    assert readiness.error_code == "unavailable"


@pytest.mark.anyio
async def test_generate_structured_sends_schema_and_validates_response() -> None:
    captured_payload: dict[str, object] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured_payload.update(json.loads(request.content))
        response = {
            "passage_index": 0,
            "summary": "Ce passage explique de manière détaillée comment préparer le bois.",
            "knowledge": [
                {
                    "title": "Préparer le bois avant vernissage",
                    "content": (
                        "Le bois doit être poncé progressivement avant l'application du vernis."
                    ),
                    "tags": ["#Bois", "Finition"],
                    "passage_indices": [0],
                }
            ],
        }
        return httpx2.Response(
            200,
            json={"response": json.dumps(response, ensure_ascii=False), "done": True},
        )

    generator = OllamaTextGenerator(_settings(), transport=httpx2.MockTransport(handler))

    result = await generator.generate_structured(
        prompt="Analyse ce passage.",
        response_model=PassageAnalysis,
        call_type="passage_analysis",
        system_prompt="Reste fidèle.",
    )

    assert result.knowledge[0].tags == ["bois", "finition"]
    assert captured_payload["model"] == "qwen3.5:4b"
    assert captured_payload["stream"] is False
    assert captured_payload["think"] is False
    assert captured_payload["format"] == PassageAnalysis.model_json_schema()
    assert captured_payload["options"] == {
        "num_ctx": 8192,
        "num_predict": 512,
        "temperature": 0.0,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("call_type", "expected_num_predict"),
    [
        ("passage_analysis", 901),
        ("hierarchical_summary", 602),
        ("final_summary", 1303),
    ],
)
async def test_all_call_types_force_think_false_and_select_their_output_limit(
    monkeypatch: pytest.MonkeyPatch,
    call_type: str,
    expected_num_predict: int,
) -> None:
    captured_payload: dict[str, object] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        captured_payload.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "passage_index": 0,
                        "summary": "Résumé fidèle suffisamment long pour valider la structure.",
                        "knowledge": [],
                    }
                ),
                "done": True,
            },
        )

    # Cette ancienne variable est volontairement ignoree : aucun deploiement ne doit
    # pouvoir reactiver le raisonnement etendu dans le pipeline Phase 3.
    monkeypatch.setenv("OLLAMA_THINK", "true")
    generator = OllamaTextGenerator(
        _settings(
            ollama_num_predict_passage_analysis=901,
            ollama_num_predict_hierarchical_summary=602,
            ollama_num_predict_final_summary=1303,
        ),
        transport=httpx2.MockTransport(handler),
    )

    await generator.generate_structured(
        prompt="DONNEES_PRIVEES_DU_PROMPT",
        response_model=PassageAnalysis,
        call_type=call_type,  # type: ignore[arg-type]
    )

    assert captured_payload["think"] is False
    assert captured_payload["options"]["num_predict"] == expected_num_predict  # type: ignore[index]


@pytest.mark.anyio
async def test_generation_logs_metrics_and_retry_without_private_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    call_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        metrics = {
            "total_duration": 5_000_000 + call_count,
            "prompt_eval_count": 120 + call_count,
            "prompt_eval_duration": 2_000_000 + call_count,
            "eval_count": 40 + call_count,
            "eval_duration": 3_000_000 + call_count,
        }
        if call_count == 1:
            return httpx2.Response(
                200,
                json={"response": "REPONSE_PRIVEE_INVALIDE", "done": True, **metrics},
            )
        return httpx2.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "passage_index": 0,
                        "summary": (
                            "REPONSE_PRIVEE_VALIDE avec un résumé assez long pour la validation."
                        ),
                        "knowledge": [],
                    }
                ),
                "done": True,
                **metrics,
            },
        )

    generator = OllamaTextGenerator(
        _settings(extraction_max_retries=1),
        transport=httpx2.MockTransport(handler),
    )
    caplog.set_level(logging.INFO, logger="second_brain.llm.client")

    captured_metrics: list[GenerationAttemptMetrics] = []
    await generator.generate_structured(
        prompt="PROMPT_SOURCE_TRES_PRIVE",
        response_model=PassageAnalysis,
        call_type="passage_analysis",
        metrics_callback=captured_metrics.append,
    )

    records = [record for record in caplog.records if record.name == "second_brain.llm.client"]
    assert len(records) == 2
    assert [record.retry for record in records] == [0, 1]  # type: ignore[attr-defined]
    assert [record.outcome for record in records] == [  # type: ignore[attr-defined]
        "validation_retry",
        "success",
    ]
    assert records[1].call_type == "passage_analysis"  # type: ignore[attr-defined]
    assert records[1].prompt_eval_count == 122  # type: ignore[attr-defined]
    assert records[1].prompt_eval_duration_ns == 2_000_002  # type: ignore[attr-defined]
    assert records[1].eval_count == 42  # type: ignore[attr-defined]
    assert records[1].eval_duration_ns == 3_000_002  # type: ignore[attr-defined]
    assert records[1].duration_seconds >= 0  # type: ignore[attr-defined]
    assert "PROMPT_SOURCE_TRES_PRIVE" not in caplog.text
    assert "REPONSE_PRIVEE_INVALIDE" not in caplog.text
    assert "REPONSE_PRIVEE_VALIDE" not in caplog.text
    assert len(captured_metrics) == 2
    assert captured_metrics[1].total_duration_ns == 5_000_002
    assert captured_metrics[1].attempt == 1


@pytest.mark.anyio
async def test_semantic_provenance_error_is_retried_with_specialized_schema() -> None:
    payloads: list[dict[str, object]] = []
    recorded_metrics: list[GenerationAttemptMetrics] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        reference = 0 if len(payloads) == 1 else 1
        return httpx2.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "passage_index": 1,
                        "summary": "Résumé fidèle et suffisamment détaillé pour ce passage.",
                        "knowledge": [
                            {
                                "title": "Connaissance correctement structurée",
                                "content": (
                                    "Cette connaissance autonome reste fidèle au passage analysé."
                                ),
                                "tags": ["test"],
                                "passage_indices": [reference],
                            }
                        ],
                    }
                ),
                "done": True,
            },
        )

    def validate_provenance(result: PassageAnalysis) -> None:
        references = result.knowledge[0].passage_indices
        if references != [1]:
            raise StructuredOutputValidationError(
                "knowledge[0].passage_indices : attendu [1], reçu [0]",
                field="knowledge[0].passage_indices",
            )

    context = GenerationCallContext(
        source_id=UUID("00000000-0000-0000-0000-000000000001"),
        processing_job_id=UUID("00000000-0000-0000-0000-000000000002"),
        passage_id=UUID("00000000-0000-0000-0000-000000000003"),
        passage_index=1,
        stage="extracting_knowledge",
    )
    generator = OllamaTextGenerator(
        _settings(extraction_max_retries=1, extraction_max_knowledge_per_passage=4),
        transport=httpx2.MockTransport(handler),
    )

    result = await generator.generate_structured(
        prompt="Analyse ce passage sans donnée privée.",
        response_model=PassageAnalysis,
        call_type="passage_analysis",
        context=context,
        metrics_callback=recorded_metrics.append,
        result_validator=validate_provenance,
    )

    assert result.knowledge[0].passage_indices == [1]
    assert len(payloads) == 2
    assert [payload["think"] for payload in payloads] == [False, False]
    assert "attendu [1], reçu [0]" in str(payloads[1]["prompt"])
    assert "Schema JSON exact" not in str(payloads[0]["prompt"])
    schema = payloads[0]["format"]
    assert schema["properties"]["passage_index"]["minimum"] == 1  # type: ignore[index]
    assert schema["properties"]["passage_index"]["maximum"] == 1  # type: ignore[index]
    assert schema["properties"]["knowledge"]["maxItems"] == 4  # type: ignore[index]
    assert [metrics.outcome for metrics in recorded_metrics] == [
        "validation_retry",
        "success",
    ]


@pytest.mark.anyio
async def test_pydantic_list_item_error_has_safe_precise_location() -> None:
    private_marker = "CONTENU_SOURCE_TRES_PRIVE"

    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "passage_index": 0,
                        "summary": "Résumé fidèle et suffisamment détaillé pour ce passage.",
                        "knowledge": [
                            {
                                "title": "Connaissance structurée",
                                "content": private_marker,
                                "tags": ["valide", 42],
                                "passage_indices": [0],
                            }
                        ],
                    }
                ),
                "done": True,
            },
        )

    generator = OllamaTextGenerator(
        _settings(extraction_max_retries=0),
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(OllamaInvalidResponseError) as captured:
        await generator.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
        )

    assert captured.value.detail is not None
    assert "knowledge[0].tags[1]" in captured.value.detail
    assert "string_type" in captured.value.detail
    assert private_marker not in captured.value.detail


@pytest.mark.anyio
async def test_semantic_validation_retries_are_strictly_exhausted() -> None:
    call_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        return httpx2.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "passage_index": 2,
                        "summary": "Résumé fidèle et suffisamment détaillé pour ce passage.",
                        "knowledge": [],
                    }
                ),
                "done": True,
            },
        )

    def reject_result(result: PassageAnalysis) -> None:
        raise StructuredOutputValidationError(
            "passage_index : attendu 1, reçu 2",
            field="passage_index",
        )

    generator = OllamaTextGenerator(
        _settings(extraction_max_retries=1),
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(OllamaInvalidResponseError) as captured:
        await generator.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
            result_validator=reject_result,
        )

    assert call_count == 2
    assert captured.value.detail == "passage_index : attendu 1, reçu 2"


@pytest.mark.anyio
async def test_invalid_json_is_retried_then_rejected() -> None:
    call_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        return httpx2.Response(200, json={"response": "not-json", "done": True})

    generator = OllamaTextGenerator(
        _settings(extraction_max_retries=1),
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(OllamaInvalidResponseError, match="Validation"):
        await generator.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
        )

    assert call_count == 2


@pytest.mark.anyio
async def test_invalid_json_can_recover_on_bounded_retry() -> None:
    call_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx2.Response(200, json={"response": "not-json", "done": True})
        return httpx2.Response(
            200,
            json={
                "response": json.dumps(
                    {
                        "passage_index": 0,
                        "summary": "Résumé corrigé, suffisamment détaillé et fidèle au passage.",
                        "knowledge": [],
                    }
                ),
                "done": True,
            },
        )

    generator = OllamaTextGenerator(
        _settings(extraction_max_retries=1),
        transport=httpx2.MockTransport(handler),
    )

    result = await generator.generate_structured(
        prompt="Analyse.",
        response_model=PassageAnalysis,
        call_type="passage_analysis",
    )

    assert result.summary == "Résumé corrigé, suffisamment détaillé et fidèle au passage."
    assert call_count == 2


@pytest.mark.anyio
async def test_model_not_found_has_actionable_error() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"error": "model 'qwen3.5:4b' not found"})

    generator = OllamaTextGenerator(_settings(), transport=httpx2.MockTransport(handler))

    with pytest.raises(OllamaModelNotFoundError, match="ollama pull qwen3.5:4b"):
        await generator.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
        )


@pytest.mark.anyio
async def test_generic_http_failure_stays_distinct_from_missing_model() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": "internal server error"})

    generator = OllamaTextGenerator(_settings(), transport=httpx2.MockTransport(handler))

    with pytest.raises(OllamaHTTPError) as error:
        await generator.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
        )

    assert error.value.status_code == 500
    assert error.value.detail == "internal server error"


@pytest.mark.anyio
async def test_generation_maps_timeout_and_connection_errors() -> None:
    async def timeout_handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("too slow", request=request)

    timed_out = OllamaTextGenerator(
        _settings(ollama_request_timeout_seconds=12),
        transport=httpx2.MockTransport(timeout_handler),
    )
    with pytest.raises(OllamaTimeoutError, match="12 secondes"):
        await timed_out.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
        )

    async def unavailable_handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    unavailable = OllamaTextGenerator(
        _settings(), transport=httpx2.MockTransport(unavailable_handler)
    )
    with pytest.raises(OllamaUnavailableError, match="indisponible"):
        await unavailable.generate_structured(
            prompt="Analyse.",
            response_model=PassageAnalysis,
            call_type="passage_analysis",
        )
