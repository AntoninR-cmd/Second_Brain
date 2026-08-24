from __future__ import annotations

import pytest
from pydantic import ValidationError
from second_brain.core.config import Settings


def test_ollama_and_chunk_defaults_are_local_and_bounded() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.ollama_generation_model == "qwen3.5:4b"
    assert settings.ollama_embedding_model == "qwen3-embedding:0.6b"
    assert settings.ollama_request_timeout_seconds == 600
    assert settings.ollama_embedding_timeout_seconds == 600
    assert settings.ollama_readiness_timeout_seconds == 5
    assert settings.ollama_num_predict_passage_analysis == 512
    assert settings.ollama_num_predict_hierarchical_summary == 512
    assert settings.ollama_num_predict_final_summary == 1024
    assert settings.ollama_num_predict_rag == 384
    assert settings.ollama_extraction_temperature == 0.0
    assert settings.ollama_rag_temperature == 0.1
    assert settings.extraction_max_retries == 1
    assert settings.extraction_max_knowledge_per_passage == 2
    assert settings.chunk_target_tokens == 800
    assert settings.chunk_max_tokens == 1200
    assert settings.embedding_batch_size == 8
    assert settings.semantic_search_top_k == 5
    assert settings.rag_retrieval_top_k == 8
    assert settings.rag_context_max_nodes == 5
    assert settings.rag_context_max_chars == 10_000


def test_plain_ollama_environment_variables_are_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999/")
    monkeypatch.setenv("OLLAMA_GENERATION_MODEL", "modele-test:1")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "embedding-test:1")
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("OLLAMA_EMBEDDING_TIMEOUT_SECONDS", "84")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT_PASSAGE_ANALYSIS", "900")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT_HIERARCHICAL_SUMMARY", "600")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT_FINAL_SUMMARY", "1200")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT_RAG", "700")
    monkeypatch.setenv("OLLAMA_EXTRACTION_TEMPERATURE", "0.05")
    monkeypatch.setenv("OLLAMA_RAG_TEMPERATURE", "0.08")
    monkeypatch.setenv("EXTRACTION_MAX_RETRIES", "2")
    monkeypatch.setenv("EXTRACTION_MAX_KNOWLEDGE_PER_PASSAGE", "3")
    monkeypatch.setenv("CHUNK_TARGET_TOKENS", "100")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "150")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("SEMANTIC_SEARCH_TOP_K", "12")
    monkeypatch.setenv("RAG_RETRIEVAL_TOP_K", "9")
    monkeypatch.setenv("RAG_CONTEXT_MAX_NODES", "7")
    monkeypatch.setenv("RAG_CONTEXT_MAX_CHARS", "18000")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:9999"
    assert settings.ollama_generation_model == "modele-test:1"
    assert settings.ollama_embedding_model == "embedding-test:1"
    assert settings.ollama_request_timeout_seconds == 42
    assert settings.ollama_embedding_timeout_seconds == 84
    assert settings.ollama_num_predict_passage_analysis == 900
    assert settings.ollama_num_predict_hierarchical_summary == 600
    assert settings.ollama_num_predict_final_summary == 1200
    assert settings.ollama_num_predict_rag == 700
    assert settings.ollama_extraction_temperature == 0.05
    assert settings.ollama_rag_temperature == 0.08
    assert settings.extraction_max_retries == 2
    assert settings.extraction_max_knowledge_per_passage == 3
    assert settings.chunk_target_tokens == 100
    assert settings.chunk_max_tokens == 150
    assert settings.embedding_batch_size == 4
    assert settings.semantic_search_top_k == 12
    assert settings.rag_retrieval_top_k == 9
    assert settings.rag_context_max_nodes == 7
    assert settings.rag_context_max_chars == 18_000


def test_chunk_target_cannot_exceed_maximum() -> None:
    with pytest.raises(ValidationError, match="CHUNK_TARGET_TOKENS"):
        Settings(_env_file=None, chunk_target_tokens=200, chunk_max_tokens=100)
