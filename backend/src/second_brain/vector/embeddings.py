from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

import httpx2

from second_brain.core.config import Settings
from second_brain.llm.errors import (
    OllamaHTTPError,
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from second_brain.llm.schemas import OllamaReadiness

EmbeddingOperation = Literal[
    "embedding",
    "index",
    "index_knowledge",
    "rebuild",
    "search",
    "semantic_search",
    "probe",
]
EmbeddingOutcome = Literal[
    "success",
    "transport_error",
    "http_error",
    "invalid_response",
]


@dataclass(frozen=True, slots=True)
class EmbeddingCallContext:
    processing_job_id: UUID | None = None
    operation: EmbeddingOperation = "embedding"
    batch_index: int | None = None
    batch_total: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingCallMetrics:
    model: str
    batch_size: int
    duration_seconds: float
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    outcome: EmbeddingOutcome


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    dimension: int
    metrics: EmbeddingCallMetrics


EmbeddingMetricsCallback = Callable[[EmbeddingCallMetrics], None]
logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    @property
    def configured_model(self) -> str:
        """Return the model used for both documents and queries."""

    async def get_readiness(self) -> OllamaReadiness:
        """Return a non-raising Ollama and embedding-model status."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        context: EmbeddingCallContext | None = None,
        metrics_callback: EmbeddingMetricsCallback | None = None,
    ) -> EmbeddingBatchResult:
        """Embed a non-empty batch while preserving input order."""


class OllamaEmbeddingProvider:
    """Small, mockable adapter around Ollama's batch embedding endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_embedding_model
        self._request_timeout = settings.ollama_embedding_timeout_seconds
        self._readiness_timeout = settings.ollama_readiness_timeout_seconds
        self._keep_alive = settings.ollama_keep_alive
        self._transport = transport

    @property
    def configured_model(self) -> str:
        return self._model

    async def get_readiness(self) -> OllamaReadiness:
        try:
            response = await self._request("GET", "/api/tags", timeout=self._readiness_timeout)
        except OllamaTimeoutError as error:
            return self._unready("timeout", error.message)
        except OllamaUnavailableError as error:
            return self._unready("unavailable", error.message)

        if response.status_code >= 400:
            return self._unready(
                "http_error",
                f"Ollama a repondu avec une erreur HTTP {response.status_code}.",
            )

        try:
            models = _read_model_names(response.json())
        except (TypeError, ValueError):
            return self._unready(
                "invalid_response",
                "Ollama a renvoye une liste de modeles illisible.",
            )

        model_available = any(_model_names_match(self._model, model) for model in models)
        if model_available:
            message = f"Ollama et le modele d'embedding {self._model} sont disponibles."
        else:
            message = (
                f"Ollama repond, mais le modele d'embedding {self._model} est absent. "
                f"Installez-le explicitement avec : ollama pull {self._model}"
            )
        return OllamaReadiness(
            ollama_available=True,
            configured_model=self._model,
            model_available=model_available,
            available_models=models,
            error_code=None,
            message=message,
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        context: EmbeddingCallContext | None = None,
        metrics_callback: EmbeddingMetricsCallback | None = None,
    ) -> EmbeddingBatchResult:
        batch = _validate_embedding_input(texts)
        started_at = perf_counter()
        try:
            response = await self._request(
                "POST",
                "/api/embed",
                timeout=self._request_timeout,
                json={
                    "model": self._model,
                    "input": batch,
                    # Silent truncation would make the stored fingerprint lie about
                    # which semantic text was actually embedded.
                    "truncate": False,
                    "keep_alive": self._keep_alive,
                },
            )
        except (OllamaTimeoutError, OllamaUnavailableError) as error:
            _record_embedding_call(
                model=self._model,
                batch_size=len(batch),
                started_at=started_at,
                response=None,
                outcome="transport_error",
                context=context,
                metrics_callback=metrics_callback,
                error_type=type(error).__name__,
            )
            raise

        try:
            self._raise_for_embedding_error(response)
        except (OllamaHTTPError, OllamaModelNotFoundError) as error:
            _record_embedding_call(
                model=self._model,
                batch_size=len(batch),
                started_at=started_at,
                response=response,
                outcome="http_error",
                context=context,
                metrics_callback=metrics_callback,
                error_type=type(error).__name__,
            )
            raise

        try:
            vectors = _read_embedding_vectors(response, expected_count=len(batch))
        except ValueError as error:
            _record_embedding_call(
                model=self._model,
                batch_size=len(batch),
                started_at=started_at,
                response=response,
                outcome="invalid_response",
                context=context,
                metrics_callback=metrics_callback,
                error_type=type(error).__name__,
            )
            raise OllamaInvalidResponseError(
                "Ollama a renvoye des embeddings invalides.",
                detail=str(error),
            ) from error

        metrics = _record_embedding_call(
            model=self._model,
            batch_size=len(batch),
            started_at=started_at,
            response=response,
            outcome="success",
            context=context,
            metrics_callback=metrics_callback,
        )
        return EmbeddingBatchResult(
            model=self._model,
            vectors=vectors,
            dimension=len(vectors[0]),
            metrics=metrics,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        json: dict[str, object] | None = None,
    ) -> httpx2.Response:
        try:
            async with httpx2.AsyncClient(
                base_url=self._base_url,
                timeout=timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                return await client.request(method, path, json=json)
        except httpx2.TimeoutException as error:
            raise OllamaTimeoutError(
                f"Ollama n'a pas repondu dans le delai de {timeout:g} secondes."
            ) from error
        except httpx2.RequestError as error:
            raise OllamaUnavailableError(
                "Ollama est indisponible. Verifiez qu'il est installe et demarre."
            ) from error

    def _raise_for_embedding_error(self, response: httpx2.Response) -> None:
        if response.status_code < 400:
            return
        detail = _response_error_detail(response)
        normalized_detail = detail.casefold()
        if (
            response.status_code in {400, 404}
            and "model" in normalized_detail
            and any(marker in normalized_detail for marker in ("not found", "missing", "absent"))
        ):
            raise OllamaModelNotFoundError(
                f"Le modele Ollama configure ({self._model}) est absent. "
                f"Installez-le explicitement avec : ollama pull {self._model}",
            )
        raise OllamaHTTPError(
            f"Ollama a repondu avec une erreur HTTP {response.status_code}.",
            status_code=response.status_code,
        )

    def _unready(self, error_code: str, message: str) -> OllamaReadiness:
        return OllamaReadiness(
            ollama_available=False,
            configured_model=self._model,
            model_available=False,
            available_models=[],
            error_code=error_code,
            message=message,
        )


def _validate_embedding_input(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)):
        raise TypeError("embed attend une sequence de textes, pas un texte unique")
    batch = list(texts)
    if not batch:
        raise ValueError("un batch d'embeddings ne peut pas etre vide")
    if any(not isinstance(text, str) or not text.strip() for text in batch):
        raise ValueError("chaque texte a encoder doit etre une chaine non vide")
    return batch


def _read_embedding_vectors(
    response: httpx2.Response,
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        payload = response.json()
    except ValueError as error:
        raise ValueError("la reponse HTTP n'est pas un objet JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("la reponse HTTP n'est pas un objet JSON")
    raw_vectors = payload.get("embeddings")
    if not isinstance(raw_vectors, list):
        raise ValueError("le champ embeddings est absent ou n'est pas une liste")
    if len(raw_vectors) != expected_count:
        raise ValueError(
            f"embeddings contient {len(raw_vectors)} vecteur(s), {expected_count} attendu(s)"
        )

    vectors: list[tuple[float, ...]] = []
    dimension: int | None = None
    for vector_index, raw_vector in enumerate(raw_vectors):
        if not isinstance(raw_vector, list) or not raw_vector:
            raise ValueError(f"embeddings[{vector_index}] doit etre un vecteur non vide")
        vector: list[float] = []
        for value_index, raw_value in enumerate(raw_vector):
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(
                    f"embeddings[{vector_index}][{value_index}] doit etre un nombre fini"
                )
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(
                    f"embeddings[{vector_index}][{value_index}] doit etre un nombre fini"
                )
            vector.append(value)
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError(
                f"embeddings[{vector_index}] a une dimension {len(vector)}, {dimension} attendue"
            )
        vectors.append(tuple(vector))
    return tuple(vectors)


def _read_model_names(payload: object) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("le champ models est absent")
    names: list[str] = []
    for item in payload["models"]:
        if not isinstance(item, dict):
            raise ValueError("une entree de modele est invalide")
        name = item.get("name") or item.get("model")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("un nom de modele est invalide")
        normalized_name = name.strip()
        if normalized_name not in names:
            names.append(normalized_name)
    return names


def _model_names_match(configured: str, available: str) -> bool:
    configured_name = configured.casefold()
    available_name = available.casefold()
    return configured_name == available_name or (
        ":" not in configured_name and available_name == f"{configured_name}:latest"
    )


def _response_error_detail(response: httpx2.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:1000]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"].strip()[:1000]
    return response.text.strip()[:1000]


def _record_embedding_call(
    *,
    model: str,
    batch_size: int,
    started_at: float,
    response: httpx2.Response | None,
    outcome: EmbeddingOutcome,
    context: EmbeddingCallContext | None,
    metrics_callback: EmbeddingMetricsCallback | None,
    error_type: str | None = None,
) -> EmbeddingCallMetrics:
    duration_seconds = perf_counter() - started_at
    raw_metrics = _read_embedding_metrics(response)
    metrics = EmbeddingCallMetrics(
        model=model,
        batch_size=batch_size,
        duration_seconds=duration_seconds,
        total_duration_ns=raw_metrics["total_duration"],
        load_duration_ns=raw_metrics["load_duration"],
        prompt_eval_count=raw_metrics["prompt_eval_count"],
        outcome=outcome,
    )
    log_method = logger.info if outcome == "success" else logger.warning
    log_method(
        "Ollama embedding processing_job_id=%s operation=%s batch_index=%s "
        "batch_total=%s model=%s batch_size=%d duration_seconds=%.3f "
        "total_duration_ns=%s load_duration_ns=%s prompt_eval_count=%s "
        "error_type=%s outcome=%s",
        context.processing_job_id if context else None,
        context.operation if context else "embedding",
        context.batch_index if context else None,
        context.batch_total if context else None,
        model,
        batch_size,
        duration_seconds,
        metrics.total_duration_ns,
        metrics.load_duration_ns,
        metrics.prompt_eval_count,
        error_type,
        outcome,
        extra={
            "processing_job_id": str(context.processing_job_id)
            if context and context.processing_job_id
            else None,
            "operation": context.operation if context else "embedding",
            "batch_index": context.batch_index if context else None,
            "batch_total": context.batch_total if context else None,
            "model": model,
            "batch_size": batch_size,
            "duration_seconds": duration_seconds,
            "total_duration_ns": metrics.total_duration_ns,
            "load_duration_ns": metrics.load_duration_ns,
            "prompt_eval_count": metrics.prompt_eval_count,
            "error_type": error_type,
            "outcome": outcome,
        },
    )
    if metrics_callback is not None:
        metrics_callback(metrics)
    return metrics


def _read_embedding_metrics(response: httpx2.Response | None) -> dict[str, int | None]:
    names = ("total_duration", "load_duration", "prompt_eval_count")
    if response is None:
        return dict.fromkeys(names)
    try:
        payload = response.json()
    except ValueError:
        return dict.fromkeys(names)
    if not isinstance(payload, dict):
        return dict.fromkeys(names)
    return {
        name: (
            value
            if isinstance((value := payload.get(name)), int) and not isinstance(value, bool)
            else None
        )
        for name in names
    }
