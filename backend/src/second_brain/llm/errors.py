from __future__ import annotations


class OllamaError(RuntimeError):
    code = "ollama_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class OllamaUnavailableError(OllamaError):
    code = "ollama_unavailable"


class OllamaTimeoutError(OllamaError):
    code = "ollama_timeout"


class OllamaModelNotFoundError(OllamaError):
    code = "ollama_model_missing"


class OllamaHTTPError(OllamaError):
    code = "ollama_http_error"

    def __init__(self, message: str, *, status_code: int, detail: str | None = None) -> None:
        super().__init__(message, detail=detail)
        self.status_code = status_code


class OllamaInvalidResponseError(OllamaError):
    code = "ollama_invalid_response"


class StructuredOutputValidationError(ValueError):
    """A semantic structured-output error containing no private source text."""

    def __init__(self, detail: str, *, field: str | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.field = field
