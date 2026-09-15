"""Template re-authoring pipeline.

Turns a canonical, ingested problem into a product-owned "template" by
prompting an LLM to rewrite it (domain swap, rephrasing, tie-breakers)
while preserving the underlying algorithmic core, then validates the
rewrite by running the canonical solver against it via `sandbox`.

Components:
  - `spec_builder`  — builds the deterministic INPUT_SPEC and renders the
                      few-shot prompt sent to the LLM.
  - `llm`           — thin, swappable LLM client: `MockLLMClient` (used
                      by all tests, no network) and `OllamaLLMClient`
                      (real HTTP path, requires a local Ollama instance).
  - `reauthor`      — orchestrates the LLM call, JSON parsing/repair loop,
                      and per-record `ReauthorResult`.
  - `validator`     — hard-rejects templates whose sample tests fail against
                      the canonical solver; soft-flags edge cases for review.
  - `storage`       — persists accepted templates and variants to SQLite.
  - `viewer`        — generates an interactive HTML dashboard and text output
                      for exploring reauthored templates.
  - `pipeline`      — batch entry point; incremental by default (skips records
                      that already have a template unless `force=True`).

Set `TRANSFORM_LLM_MODE=ollama` (and `TRANSFORM_LLM_MODEL`) to use a real
local model. Defaults to the mock client, which cleanly rejects everything
(useful for pipeline-wiring tests without a running model).
"""
from .llm import LLMResponse, MockLLMClient, OllamaLLMClient, get_default_client

__all__ = ["LLMResponse", "MockLLMClient", "OllamaLLMClient", "get_default_client"]
