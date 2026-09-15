"""LLM wrapper: a thin, swappable client for calling a local instruction
model (e.g. Mistral-Instruct-7B via Ollama) to re-author problems.

STATUS: `MockLLMClient` is complete and used by all current tests — it
returns deterministic canned responses with zero network calls, so
`transform/` code can be built and tested without any model installed.
`OllamaLLMClient` implements the real HTTP call path but has NOT been
tested against a running model in this environment (no local model
runtime was available/installed at the time this was written — `ollama`
is not on PATH and nothing responds on localhost:11434). Treat it as
"should work per Ollama's documented API shape," not "verified."

--------------------------------------------------------------------
Running a real local model later (Mistral-Instruct-7B via Ollama)
--------------------------------------------------------------------
1. Install Ollama: https://ollama.com/download
2. Pull an instruction-tuned Mistral model, e.g.:
       ollama pull mistral:instruct
   (or a specific quantized tag if you want to control VRAM/RAM usage,
   e.g. `mistral:7b-instruct-q4_0` for a smaller footprint on CPU)
3. Ollama runs its API automatically once installed
   (http://localhost:11434 by default — no extra "ollama serve" step
   needed on most installs, but run it manually if the API isn't
   reachable).
4. Point this module at it:
       export TRANSFORM_LLM_MODE=ollama
       export TRANSFORM_LLM_MODEL=mistral:instruct   # match the pulled tag
   Then `get_default_client()` returns an `OllamaLLMClient` instead of
   the mock. Everything else in `transform/` is written against the
   `LLMClient` protocol below, so no other code needs to change.
5. Sanity check the endpoint directly before trusting this wrapper:
       curl http://localhost:11434/api/generate -d '{
         "model": "mistral:instruct", "prompt": "hello", "stream": false
       }'

Fallback model (per the design doc, if you have >40GB VRAM and want
higher quality generation): Llama 2 13B, same Ollama flow
(`ollama pull llama2:13b`), same env vars.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral:instruct"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0


class LLMCallError(RuntimeError):
    """Raised when an LLM call fails after retries (connection error,
    non-2xx HTTP status, or a response shape the client can't parse)."""


@dataclass
class LLMResponse:
    text: str
    model: str
    mock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    duration_seconds: float = 0.0


class LLMClient(Protocol):
    """Structural interface every LLM client in this module satisfies.
    `transform/` orchestration code should type against this, not a
    concrete client, so swapping mock <-> real Ollama <-> some other
    backend later needs zero call-site changes.
    """

    def call(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> LLMResponse: ...


class MockLLMClient:
    """Deterministic, network-free stand-in for a real LLM.

    Two ways to control what it returns, checked in this order:
      1. `responses`: an explicit list consumed in call order (useful for
         testing retry/re-prompt logic — e.g. first response malformed,
         second valid).
      2. `response_fn`: a callable `(prompt, system) -> str`, for tests
         that want to react to the actual prompt content.
      3. Falls back to `default_response` (a fixed string) forever.

    Every call is recorded in `.calls` for test assertions.
    """

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        response_fn: Optional[Callable[[str, Optional[str]], str]] = None,
        default_response: str = '{"error": "mock client has no configured response"}',
        model: str = "mock-llm",
    ):
        self._responses = list(responses) if responses else None
        self._response_fn = response_fn
        self._default_response = default_response
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def call(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> LLMResponse:
        self.calls.append(
            {"prompt": prompt, "system": system, "temperature": temperature, "max_tokens": max_tokens}
        )

        if self._responses:
            # Consume in order; once exhausted, keep returning the last one
            # rather than raising IndexError — re-prompt loops that retry
            # more times than there are canned responses shouldn't crash.
            idx = min(len(self.calls) - 1, len(self._responses) - 1)
            text = self._responses[idx]
        elif self._response_fn is not None:
            text = self._response_fn(prompt, system)
        else:
            text = self._default_response

        return LLMResponse(text=text, model=self.model, mock=True, attempts=1, duration_seconds=0.0)


class OllamaLLMClient:
    """Real HTTP client for Ollama's `/api/generate` endpoint.

    Uses stdlib `urllib` rather than `requests` since `requests` isn't a
    declared dependency of this project and adding it just for an
    unverified code path isn't worth it yet — swap this for `requests`
    or `httpx` later if the retry/streaming logic here needs to grow.

    NOT verified against a live Ollama instance in this environment. If
    Ollama's actual response shape differs from what's assumed below
    (`{"response": "..."}`, matching Ollama's documented non-streaming
    shape as of writing), `_extract_text` will raise `LLMCallError` with
    the raw payload attached rather than silently returning garbage —
    check that error message first if this breaks in practice.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        think: bool = False,
    ):
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.think = think  # False = disable chain-of-thought for thinking models (e.g. qwen3, deepseek-r1)

    def _extract_text(self, payload: dict[str, Any]) -> str:
        # Ollama's documented non-streaming /api/generate shape.
        if "response" in payload:
            return payload["response"]
        # Some other local-model HTTP servers (text-generation-webui,
        # older Ollama versions) use one of these instead.
        if "text" in payload:
            return payload["text"]
        if "results" in payload and payload["results"]:
            first = payload["results"][0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        raise LLMCallError(f"Unrecognized LLM response shape, no known text field found: {payload!r}")

    def call(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": temperature,
            "num_predict": max_tokens,  # Ollama's actual field name (max_tokens is silently ignored)
            "stream": False,
        }
        if not self.think:
            payload["think"] = False  # disable <think>...</think> for reasoning models
        body = json.dumps(payload).encode("utf-8")

        last_error: Optional[Exception] = None
        start = time.monotonic()
        for attempt in range(1, self.max_retries + 2):
            try:
                request = urllib.request.Request(
                    self.base_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                text = self._extract_text(raw)
                return LLMResponse(
                    text=text,
                    model=self.model,
                    mock=False,
                    raw=raw,
                    attempts=attempt,
                    duration_seconds=time.monotonic() - start,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                logger.warning(
                    "LLM call to %s failed (attempt %d/%d): %s",
                    self.base_url,
                    attempt,
                    self.max_retries + 1,
                    exc,
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
            except json.JSONDecodeError as exc:
                raise LLMCallError(f"LLM at {self.base_url} returned non-JSON response: {exc}") from exc

        raise LLMCallError(
            f"LLM call to {self.base_url} failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error


def get_default_client(think: bool = False) -> LLMClient:
    """Selects a client from environment variables, defaulting to the
    mock client since no local model is guaranteed to be running.

    Env vars:
      TRANSFORM_LLM_MODE:  "mock" (default) or "ollama"
      TRANSFORM_LLM_MODEL: model tag to request, default "mistral:instruct"
      TRANSFORM_LLM_URL:   Ollama endpoint, default http://localhost:11434/api/generate
      TRANSFORM_LLM_THINK: "true" / "1" to enable chain-of-thought for thinking
                           models (qwen3, deepseek-r1, etc.). Default "false".
    """
    mode = os.environ.get("TRANSFORM_LLM_MODE", "mock").strip().lower()
    if mode == "ollama":
        env_think = os.environ.get("TRANSFORM_LLM_THINK", "false").strip().lower()
        effective_think = think or env_think in ("true", "1", "yes")
        return OllamaLLMClient(
            base_url=os.environ.get("TRANSFORM_LLM_URL", DEFAULT_OLLAMA_URL),
            model=os.environ.get("TRANSFORM_LLM_MODEL", DEFAULT_MODEL),
            think=effective_think,
        )
    if mode != "mock":
        logger.warning("Unknown TRANSFORM_LLM_MODE=%r, falling back to mock client", mode)
    return MockLLMClient()
