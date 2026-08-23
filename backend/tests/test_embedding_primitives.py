from __future__ import annotations

import json
import logging
from hashlib import sha256

import httpx2
import pytest
from second_brain.core.config import Settings
from second_brain.llm.errors import (
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from second_brain.vector.embeddings import (
    EmbeddingCallContext,
    OllamaEmbeddingProvider,
)
from second_brain.vector.semantic_text import (
    SEMANTIC_TEXT_VERSION,
    build_semantic_text,
    semantic_text_fingerprint,
)


def _settings(**updates: object) -> Settings:
    return Settings(_env_file=None).model_copy(update=updates)


def test_semantic_text_and_fingerprint_contain_only_title_and_content() -> None:
    semantic_text = build_semantic_text(
        title="  Preparation du plastique  ",
        content="\nLe support doit etre propre.\n",
    )

    assert SEMANTIC_TEXT_VERSION == "title-content-v1"
    assert semantic_text == "Preparation du plastique\n\nLe support doit etre propre."
    assert (
        semantic_text_fingerprint(
            title="  Preparation du plastique  ",
            content="\nLe support doit etre propre.\n",
        )
        == sha256(semantic_text.encode("utf-8")).hexdigest()
    )
    assert len(semantic_text_fingerprint(title="Titre", content="Contenu")) == 64


@pytest.mark.anyio
async def test_embedding_readiness_reports_the_embedding_model_separately() -> None:
    async def available_handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/tags"
        return httpx2.Response(
            200,
            json={"models": [{"name": "qwen3.5:4b"}, {"name": "qwen3-embedding:0.6b"}]},
        )

    provider = OllamaEmbeddingProvider(
        _settings(),
        transport=httpx2.MockTransport(available_handler),
    )
    readiness = await provider.get_readiness()

    assert readiness.ollama_available is True
    assert readiness.configured_model == "qwen3-embedding:0.6b"
    assert readiness.model_available is True

    async def missing_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"models": [{"name": "qwen3.5:4b"}]})

    missing = OllamaEmbeddingProvider(
        _settings(),
        transport=httpx2.MockTransport(missing_handler),
    )
    missing_readiness = await missing.get_readiness()
    assert missing_readiness.ollama_available is True
    assert missing_readiness.model_available is False
    assert "ollama pull qwen3-embedding:0.6b" in missing_readiness.message


@pytest.mark.anyio
async def test_embedding_batch_uses_one_request_and_records_safe_metrics(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_marker = "contenu-prive-ne-jamais-journaliser"
    captured_payload: dict[str, object] = {}

    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/api/embed"
        captured_payload.update(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "qwen3-embedding:0.6b",
                "embeddings": [[1, 0.25, -0.5], [0.1, 0.2, 0.3]],
                "total_duration": 4_000_000,
                "load_duration": 1_000_000,
                "prompt_eval_count": 18,
            },
        )

    metrics = []
    provider = OllamaEmbeddingProvider(
        _settings(ollama_embedding_timeout_seconds=42),
        transport=httpx2.MockTransport(handler),
    )
    with caplog.at_level(logging.INFO, logger="second_brain.vector.embeddings"):
        result = await provider.embed(
            [private_marker, "deuxieme texte"],
            context=EmbeddingCallContext(operation="index_knowledge", batch_index=1, batch_total=3),
            metrics_callback=metrics.append,
        )

    assert captured_payload == {
        "model": "qwen3-embedding:0.6b",
        "input": [private_marker, "deuxieme texte"],
        "truncate": False,
        "keep_alive": "5m",
    }
    assert result.model == "qwen3-embedding:0.6b"
    assert result.dimension == 3
    assert result.vectors == ((1.0, 0.25, -0.5), (0.1, 0.2, 0.3))
    assert result.metrics.total_duration_ns == 4_000_000
    assert result.metrics.load_duration_ns == 1_000_000
    assert result.metrics.prompt_eval_count == 18
    assert metrics == [result.metrics]
    assert private_marker not in caplog.text
    assert "[1.0" not in caplog.text
    assert "batch_size=2" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("embeddings", "expected_detail"),
    [
        ([], "0 vecteur"),
        ([[1.0, 2.0]], "1 vecteur"),
        ([[1.0, 2.0], []], "vecteur non vide"),
        ([[1.0, 2.0], [3.0]], "dimension 1"),
        ([[1.0, 2.0], [3.0, float("nan")]], "nombre fini"),
        ([[1.0, 2.0], [3.0, "4"]], "nombre fini"),
    ],
)
async def test_embedding_response_validation_is_strict_and_diagnostic(
    embeddings: object,
    expected_detail: str,
) -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if "nan" in repr(embeddings):
            return httpx2.Response(
                200,
                content=b'{"embeddings":[[1.0,2.0],[3.0,NaN]]}',
                headers={"Content-Type": "application/json"},
            )
        return httpx2.Response(200, json={"embeddings": embeddings})

    provider = OllamaEmbeddingProvider(
        _settings(),
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(OllamaInvalidResponseError) as raised:
        await provider.embed(["premier", "second"])

    assert expected_detail in (raised.value.detail or "")


@pytest.mark.anyio
async def test_embedding_model_missing_and_timeout_use_existing_ollama_errors() -> None:
    async def missing_handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"error": "model not found"})

    missing = OllamaEmbeddingProvider(
        _settings(),
        transport=httpx2.MockTransport(missing_handler),
    )
    with pytest.raises(OllamaModelNotFoundError) as missing_error:
        await missing.embed(["texte"])
    assert "ollama pull qwen3-embedding:0.6b" in missing_error.value.message

    async def timeout_handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timeout", request=request)

    timeout = OllamaEmbeddingProvider(
        _settings(ollama_embedding_timeout_seconds=17),
        transport=httpx2.MockTransport(timeout_handler),
    )
    with pytest.raises(OllamaTimeoutError) as timeout_error:
        await timeout.embed(["texte"])
    assert "17 secondes" in timeout_error.value.message


@pytest.mark.anyio
async def test_embedding_input_must_be_a_non_empty_batch() -> None:
    calls = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(200, json={"embeddings": [[1.0]]})

    provider = OllamaEmbeddingProvider(
        _settings(),
        transport=httpx2.MockTransport(handler),
    )

    with pytest.raises(ValueError):
        await provider.embed([])
    with pytest.raises(ValueError):
        await provider.embed(["   "])
    with pytest.raises(TypeError):
        await provider.embed("texte unique")
    assert calls == 0
