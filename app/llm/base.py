"""Provider interface + error taxonomy. Providers translate vendor exceptions into these, so the rest of
the system only knows 'retryable or not'."""
from typing import Protocol

from app.config import settings
from app.errors import PipelineError


class LLMError(Exception):
    retryable = False
    code = "LLM_ERROR"

    def __init__(self, message="", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMTimeout(LLMError):
    retryable, code = True, "LLM_TIMEOUT"


class LLMRateLimit(LLMError):
    retryable, code = True, "LLM_RATE_LIMITED"


class LLMUnavailable(LLMError):
    retryable, code = True, "LLM_UNAVAILABLE"


class LLMAuthError(LLMError):
    retryable, code = False, "LLM_AUTH"


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, *, purpose: str, system: str, user: str, max_tokens: int = 1024, timeout: float = 20.0) -> str: ...


def get_provider(name: str | None = None):
    name = (name or settings.llm_provider).lower()
    if name == "mock":
        from app.llm.mock import MockProvider
        return MockProvider()
    if name == "anthropic":
        if not settings.anthropic_api_key:
            raise PipelineError("LLM_NOT_CONFIGURED", "ANTHROPIC_API_KEY is not set; use LLM_PROVIDER=mock for keyless runs.")
        from app.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
    raise PipelineError("LLM_NOT_CONFIGURED", "Unknown LLM provider.", {"provider": name[:20], "allowed": ["mock", "anthropic"]})
