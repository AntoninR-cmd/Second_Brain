from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol, TypeVar
from uuid import UUID

import httpx2
from pydantic import BaseModel, ValidationError

from second_brain.core.config import Settings
from second_brain.llm.errors import (
    OllamaHTTPError,
    OllamaInvalidResponseError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    StructuredOutputValidationError,
)
from second_brain.llm.prompt_loader import build_validation_retry_prompt
from second_brain.llm.schemas import OllamaReadiness

SchemaT = TypeVar("SchemaT", bound=BaseModel)
GenerationCallType = Literal[
    "passage_analysis",
    "hierarchical_summary",
    "final_summary",
]
GenerationOutcome = Literal[
    "success",
    "validation_retry",
    "invalid_response",
    "transport_error",
    "http_error",
]


@dataclass(frozen=True, slots=True)
class GenerationCallContext:
    source_id: UUID
    processing_job_id: UUID
    stage: str
    passage_id: UUID | None = None
    passage_index: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationAttemptMetrics:
    call_type: GenerationCallType
    attempt: int
    duration_seconds: float
    total_duration_ns: int | None
    prompt_eval_count: int | None
    prompt_eval_duration_ns: int | None
    eval_count: int | None
    eval_duration_ns: int | None
    outcome: GenerationOutcome


GenerationMetricsCallback = Callable[[GenerationAttemptMetrics], None]

logger = logging.getLogger(__name__)


class TextGenerator(Protocol):
    async def get_readiness(self) -> OllamaReadiness:
        """Return a non-raising status suitable for the application UI."""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[SchemaT],
        call_type: GenerationCallType,
        system_prompt: str | None = None,
        context: GenerationCallContext | None = None,
        metrics_callback: GenerationMetricsCallback | None = None,
        result_validator: Callable[[SchemaT], None] | None = None,
    ) -> SchemaT:
        """Generate and validate one response against ``response_model``."""


class OllamaTextGenerator:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.ollama_generation_model
        self._request_timeout = settings.ollama_request_timeout_seconds
        self._readiness_timeout = settings.ollama_readiness_timeout_seconds
        self._num_ctx = settings.ollama_num_ctx
        self._temperature: dict[GenerationCallType, float] = {
            "passage_analysis": settings.ollama_extraction_temperature,
            "hierarchical_summary": settings.ollama_temperature,
            "final_summary": settings.ollama_temperature,
        }
        self._keep_alive = settings.ollama_keep_alive
        self._validation_retries = settings.extraction_max_retries
        self._max_knowledge_per_passage = settings.extraction_max_knowledge_per_passage
        self._num_predict: dict[GenerationCallType, int] = {
            "passage_analysis": settings.ollama_num_predict_passage_analysis,
            "hierarchical_summary": settings.ollama_num_predict_hierarchical_summary,
            "final_summary": settings.ollama_num_predict_final_summary,
        }
        self._transport = transport

    @property
    def configured_model(self) -> str:
        return self._model

    async def get_readiness(self) -> OllamaReadiness:
        try:
            response = await self._request("GET", "/api/tags", timeout=self._readiness_timeout)
        except OllamaTimeoutError as exc:
            return self._unready("timeout", exc.message)
        except OllamaUnavailableError as exc:
            return self._unready("unavailable", exc.message)

        if response.status_code >= 400:
            detail = _response_error_detail(response)
            return self._unready(
                "http_error",
                f"Ollama a repondu avec une erreur HTTP {response.status_code}.",
                detail=detail,
            )

        try:
            payload = response.json()
            models = _read_model_names(payload)
        except (TypeError, ValueError) as exc:
            return self._unready(
                "invalid_response",
                "Ollama a renvoye une liste de modeles illisible.",
                detail=str(exc),
            )

        model_available = any(_model_names_match(self._model, candidate) for candidate in models)
        if model_available:
            message = f"Ollama et le modele {self._model} sont disponibles."
        else:
            message = (
                f"Ollama repond, mais le modele {self._model} est absent. "
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

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[SchemaT],
        call_type: GenerationCallType,
        system_prompt: str | None = None,
        context: GenerationCallContext | None = None,
        metrics_callback: GenerationMetricsCallback | None = None,
        result_validator: Callable[[SchemaT], None] | None = None,
    ) -> SchemaT:
        schema = response_model.model_json_schema()
        if call_type == "passage_analysis" and context and context.passage_index is not None:
            _specialize_passage_schema(
                schema,
                passage_index=context.passage_index,
                max_knowledge=self._max_knowledge_per_passage,
            )
        # Le schéma est déjà transmis via `format`. Le recopier intégralement dans le
        # prompt doublait plusieurs centaines de tokens d'entrée à chaque passage.
        grounded_prompt = (
            f"{prompt.rstrip()}\n\n"
            "Respecte strictement le schéma JSON transmis par le paramètre format."
        )
        current_prompt = grounded_prompt
        last_detail: str | None = None

        for attempt in range(self._validation_retries + 1):
            started_at = perf_counter()
            try:
                response = await self._request(
                    "POST",
                    "/api/generate",
                    timeout=self._request_timeout,
                    json={
                        "model": self._model,
                        "prompt": current_prompt,
                        "system": system_prompt or "",
                        "stream": False,
                        # Le raisonnement etendu est volontairement interdit pour ce pipeline
                        # local : aucun appelant ne peut le reactiver par inadvertance.
                        "think": False,
                        "format": schema,
                        "keep_alive": self._keep_alive,
                        "options": {
                            "num_ctx": self._num_ctx,
                            "num_predict": self._num_predict[call_type],
                            "temperature": self._temperature[call_type],
                        },
                    },
                )
            except (OllamaTimeoutError, OllamaUnavailableError) as exc:
                _record_generation_attempt(
                    call_type=call_type,
                    attempt=attempt,
                    started_at=started_at,
                    response=None,
                    outcome="transport_error",
                    context=context,
                    metrics_callback=metrics_callback,
                    error_type=type(exc).__name__,
                    diagnostic=exc.message,
                )
                raise

            try:
                self._raise_for_generation_error(response)
            except (OllamaHTTPError, OllamaModelNotFoundError) as exc:
                _record_generation_attempt(
                    call_type=call_type,
                    attempt=attempt,
                    started_at=started_at,
                    response=response,
                    outcome="http_error",
                    context=context,
                    metrics_callback=metrics_callback,
                    error_type=type(exc).__name__,
                    diagnostic=exc.message,
                )
                raise

            validation_detail: str | None = None
            validation_error_type: str | None = None
            result: SchemaT | None = None
            try:
                generated_text = _read_generated_text(response)
            except ValueError as exc:
                validation_detail = str(exc)
                validation_error_type = type(exc).__name__
            else:
                try:
                    result = response_model.model_validate_json(generated_text)
                except ValidationError as exc:
                    validation_detail = _format_pydantic_validation_error(exc)
                    validation_error_type = type(exc).__name__
                else:
                    if result_validator is not None:
                        try:
                            result_validator(result)
                        except StructuredOutputValidationError as exc:
                            validation_detail = exc.detail
                            validation_error_type = type(exc).__name__

            if validation_detail is not None:
                last_detail = validation_detail
                if attempt >= self._validation_retries:
                    _record_generation_attempt(
                        call_type=call_type,
                        attempt=attempt,
                        started_at=started_at,
                        response=response,
                        outcome="invalid_response",
                        context=context,
                        metrics_callback=metrics_callback,
                        error_type=validation_error_type,
                        diagnostic=validation_detail,
                    )
                    break
                _record_generation_attempt(
                    call_type=call_type,
                    attempt=attempt,
                    started_at=started_at,
                    response=response,
                    outcome="validation_retry",
                    context=context,
                    metrics_callback=metrics_callback,
                    error_type=validation_error_type,
                    diagnostic=validation_detail,
                )
                current_prompt = build_validation_retry_prompt(
                    original_prompt=grounded_prompt,
                    validation_error=validation_detail,
                )
                continue

            if result is None:
                raise RuntimeError("La validation structurée n'a produit aucun résultat.")
            _record_generation_attempt(
                call_type=call_type,
                attempt=attempt,
                started_at=started_at,
                response=response,
                outcome="success",
                context=context,
                metrics_callback=metrics_callback,
            )
            return result

        raise OllamaInvalidResponseError(
            "Validation de la réponse Ollama impossible après les tentatives autorisées.",
            detail=last_detail,
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
        except httpx2.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama n'a pas repondu dans le delai de {timeout:g} secondes."
            ) from exc
        except httpx2.RequestError as exc:
            raise OllamaUnavailableError(
                "Ollama est indisponible. Verifiez qu'il est installe et demarre.",
                detail=str(exc),
            ) from exc

    def _raise_for_generation_error(self, response: httpx2.Response) -> None:
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
                detail=detail,
            )
        raise OllamaHTTPError(
            f"Ollama a repondu avec une erreur HTTP {response.status_code}.",
            status_code=response.status_code,
            detail=detail,
        )

    def _unready(
        self,
        error_code: str,
        message: str,
        *,
        detail: str | None = None,
    ) -> OllamaReadiness:
        if detail:
            message = f"{message} Detail : {detail}"
        return OllamaReadiness(
            ollama_available=False,
            configured_model=self._model,
            model_available=False,
            available_models=[],
            error_code=error_code,
            message=message,
        )


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


def _specialize_passage_schema(
    schema: dict[str, object],
    *,
    passage_index: int,
    max_knowledge: int,
) -> None:
    """Constrain the generic schema with the provenance known by the backend."""

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    passage_property = properties.get("passage_index")
    if isinstance(passage_property, dict):
        passage_property["minimum"] = passage_index
        passage_property["maximum"] = passage_index

    knowledge_property = properties.get("knowledge")
    if not isinstance(knowledge_property, dict):
        return
    knowledge_property["maxItems"] = max_knowledge
    item_schema = knowledge_property.get("items")
    if not isinstance(item_schema, dict):
        return
    reference = item_schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        return
    definitions = schema.get("$defs")
    if not isinstance(definitions, dict):
        return
    draft_schema = definitions.get(reference.removeprefix("#/$defs/"))
    if not isinstance(draft_schema, dict):
        return
    draft_properties = draft_schema.get("properties")
    if not isinstance(draft_properties, dict):
        return
    provenance = draft_properties.get("passage_indices")
    if not isinstance(provenance, dict):
        return
    provenance["minItems"] = 1
    provenance["maxItems"] = 1
    provenance_items = provenance.get("items")
    if isinstance(provenance_items, dict):
        provenance_items["minimum"] = passage_index
        provenance_items["maximum"] = passage_index


def _model_names_match(configured: str, available: str) -> bool:
    configured_name = configured.casefold()
    available_name = available.casefold()
    if configured_name == available_name:
        return True
    return ":" not in configured_name and available_name == f"{configured_name}:latest"


def _read_generated_text(response: httpx2.Response) -> str:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("la reponse HTTP Ollama n'est pas du JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("la reponse HTTP Ollama n'est pas un objet")
    generated_text = payload.get("response")
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise ValueError("le champ response est absent ou vide")
    if payload.get("done") is not True:
        raise ValueError("la generation Ollama n'est pas terminee")
    return generated_text


def _response_error_detail(response: httpx2.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:1000]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"].strip()[:1000]
    return response.text.strip()[:1000]


def _record_generation_attempt(
    *,
    call_type: GenerationCallType,
    attempt: int,
    started_at: float,
    response: httpx2.Response | None,
    outcome: GenerationOutcome,
    context: GenerationCallContext | None,
    metrics_callback: GenerationMetricsCallback | None,
    error_type: str | None = None,
    diagnostic: str | None = None,
) -> GenerationAttemptMetrics:
    duration_seconds = perf_counter() - started_at
    raw_metrics = _read_generation_metrics(response)
    metrics = GenerationAttemptMetrics(
        call_type=call_type,
        attempt=attempt,
        duration_seconds=duration_seconds,
        total_duration_ns=raw_metrics["total_duration"],
        prompt_eval_count=raw_metrics["prompt_eval_count"],
        prompt_eval_duration_ns=raw_metrics["prompt_eval_duration"],
        eval_count=raw_metrics["eval_count"],
        eval_duration_ns=raw_metrics["eval_duration"],
        outcome=outcome,
    )
    safe_diagnostic = diagnostic.strip()[:2000] if diagnostic else None
    log_method = logger.info if outcome == "success" else logger.warning
    log_method(
        "Ollama generation source_id=%s processing_job_id=%s passage_id=%s "
        "passage_index=%s stage=%s call_type=%s attempt=%d retry=%d "
        "duration_seconds=%.3f total_duration_ns=%s prompt_eval_count=%s "
        "prompt_eval_duration_ns=%s eval_count=%s eval_duration_ns=%s "
        "error_type=%s diagnostic=%s outcome=%s",
        context.source_id if context else None,
        context.processing_job_id if context else None,
        context.passage_id if context else None,
        context.passage_index if context else None,
        context.stage if context else None,
        call_type,
        attempt + 1,
        attempt,
        duration_seconds,
        metrics.total_duration_ns,
        metrics.prompt_eval_count,
        metrics.prompt_eval_duration_ns,
        metrics.eval_count,
        metrics.eval_duration_ns,
        error_type,
        safe_diagnostic,
        outcome,
        extra={
            "source_id": str(context.source_id) if context else None,
            "processing_job_id": str(context.processing_job_id) if context else None,
            "passage_id": str(context.passage_id) if context and context.passage_id else None,
            "passage_index": context.passage_index if context else None,
            "stage": context.stage if context else None,
            "call_type": call_type,
            "attempt": attempt + 1,
            "duration_seconds": duration_seconds,
            "total_duration_ns": metrics.total_duration_ns,
            "prompt_eval_count": metrics.prompt_eval_count,
            "prompt_eval_duration_ns": metrics.prompt_eval_duration_ns,
            "eval_count": metrics.eval_count,
            "eval_duration_ns": metrics.eval_duration_ns,
            "retry": attempt,
            "error_type": error_type,
            "diagnostic": safe_diagnostic,
            "outcome": outcome,
        },
    )
    if metrics_callback is not None:
        metrics_callback(metrics)
    return metrics


def _read_generation_metrics(
    response: httpx2.Response | None,
) -> dict[str, int | None]:
    names = (
        "total_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
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


def _format_pydantic_validation_error(error: ValidationError) -> str:
    issues: list[str] = []
    raw_issues = error.errors(include_input=False, include_url=False)
    for issue in raw_issues[:12]:
        location = _format_validation_location(issue.get("loc", ()))
        issue_type = str(issue.get("type", "validation_error"))
        message = str(issue.get("msg", "valeur invalide"))
        issues.append(f"{location}: {message} (type={issue_type})")
    remaining = len(raw_issues) - len(issues)
    if remaining > 0:
        issues.append(f"{remaining} erreur(s) supplémentaire(s) non affichée(s)")
    return "; ".join(issues) or "Réponse structurée invalide."


def _format_validation_location(location: object) -> str:
    if not isinstance(location, (tuple, list)) or not location:
        return "réponse"
    result = ""
    for part in location:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            separator = "." if result else ""
            result += f"{separator}{part}"
    return result
