"""Anthropic Messages API adapter. Retries are handled by our own resilience layer (SDK max_retries=0),
vendor exceptions are mapped to the LLMError taxonomy by class name (so tests can inject a fake client)."""
from app.llm.base import LLMAuthError, LLMError, LLMRateLimit, LLMTimeout, LLMUnavailable


def map_error(e: Exception) -> LLMError:
    n = type(e).__name__
    if n == "APITimeoutError":
        return LLMTimeout("timeout")
    if n == "RateLimitError":
        ra = None
        try:
            ra = float(getattr(getattr(e, "response", None), "headers", {}).get("retry-after"))
        except (TypeError, ValueError):
            pass
        return LLMRateLimit("rate limited", retry_after=ra)
    if n in ("APIConnectionError", "InternalServerError", "OverloadedError", "ServiceUnavailableError"):
        return LLMUnavailable(n)
    if n in ("AuthenticationError", "PermissionDeniedError"):
        return LLMAuthError(n)
    if getattr(e, "status_code", 0) and e.status_code >= 500:
        return LLMUnavailable(n)
    return LLMError(n)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, client=None):
        self.model = model
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self.client = client

    def complete(self, *, purpose: str, system: str, user: str, max_tokens: int = 1024, timeout: float = 20.0) -> str:
        try:
            resp = self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=0, system=system,
                                               messages=[{"role": "user", "content": user}], timeout=timeout)
        except Exception as e:   # noqa: BLE001 - vendor errors are mapped, never leaked
            raise map_error(e) from None
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
