"""Template re-authoring module (early scaffold).

Turns a canonical, ingested problem into a product-owned "template" by
prompting an LLM to rewrite it (domain swap, rephrasing, tie-breakers)
while preserving the underlying algorithmic core, then validating the
rewrite by running the canonical solver against it via `sandbox`.

Current state: LLM wrapper (`llm.py`) with a working mock mode and an
untested-but-real Ollama HTTP path. The rest of the pipeline described in
the design doc (spec_builder, reauthor orchestrator, variant generation,
DB tables) is not built yet — see each module's docstring for exact
status. Build order intentionally follows the priority plan: get the
LLM call point solid and testable first, since everything else composes
on top of it.
"""
from .llm import LLMResponse, MockLLMClient, OllamaLLMClient, get_default_client

__all__ = ["LLMResponse", "MockLLMClient", "OllamaLLMClient", "get_default_client"]
