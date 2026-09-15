"""Reauthor orchestrator: the control flow tying `spec_builder` -> LLM ->
JSON parsing/repair -> `validator` together for one canonical record.

Control flow (matches the design doc's pseudocode):
  1. Build the deterministic spec + prompt.
  2. Call the LLM. Parse the response as `TemplateOutput`.
  3. If parsing fails, re-prompt (up to `max_attempts`) with a
     "fix your JSON" instruction that includes the raw response and the
     parse error.
  4. Once parsed, run `validator.validate_template` against the
     canonical solution.
  5. If validation fails outright (sample tests don't reproduce), reject
     — this is a proven algorithmic mismatch, never persisted, never
     retried automatically (a different LLM sample at the same
     temperature=0 seed would just reproduce the same output anyway;
     retrying here would only help if the LLM were resampled at a
     different temperature, which is a deliberate choice left to the
     caller, not automatic).
  6. If validation passes but raised soft concerns (schema shape change,
     missing tie-breaker, some generated variants erroring), accept but
     flag `needs_review`.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from .llm import LLMClient, LLMCallError
from .models import Provenance, ReauthorResult, Template, TemplateOutput
from .spec_builder import build_json_fix_prompt, build_spec, render_prompt, SYSTEM_PROMPT
from .validator import DEFAULT_VARIANT_COUNT, validate_template

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_block(text: str) -> str:
    """LLMs frequently wrap JSON in markdown fences or add a leading/
    trailing sentence despite instructions not to. Before giving up and
    re-prompting, try extracting the first `{...}` block (greedy, since
    nested braces are common in JSON) and parsing that instead of the
    raw text verbatim.
    """
    text = text.strip()
    # Strip common markdown code fences first.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else text


def parse_template_output(raw_text: str) -> tuple[Optional[TemplateOutput], Optional[str]]:
    """Attempts to parse `raw_text` as a `TemplateOutput`. Returns
    (template_output, None) on success, or (None, error_message) on
    failure — never raises, since this sits in a retry loop.
    """
    candidate = extract_json_block(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"

    if not isinstance(payload, dict):
        return None, f"expected a JSON object, got {type(payload).__name__}"

    if "error" in payload and len(payload) == 1:
        return None, f"model declared it could not produce a template: {payload['error']}"

    try:
        return TemplateOutput.model_validate(payload), None
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, but keep this loop-safe
        return None, f"schema validation failed: {exc}"


def reauthor_template(
    canonical_record: dict[str, Any],
    llm_client: LLMClient,
    pattern_label: Optional[str] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    variant_count: int = DEFAULT_VARIANT_COUNT,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 300.0,
) -> ReauthorResult:
    """Runs the full reauthor control flow for one canonical record.
    Never raises for expected failure modes (malformed JSON after
    retries, failed validation) — those are represented in the returned
    `ReauthorResult.status`. Only raises if the LLM call itself fails
    after its own internal retries (`LLMCallError` propagates), since
    that's an infrastructure problem the caller should decide how to
    handle (skip this record vs abort the whole batch), not something
    this function can meaningfully paper over.
    """
    canonical_id = canonical_record.get("id", "")
    spec = build_spec(canonical_record, pattern_label=pattern_label)
    prompt = render_prompt(spec)

    template_output: Optional[TemplateOutput] = None
    parse_error: Optional[str] = None
    raw_response = ""
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        current_prompt = prompt if attempt == 1 else build_json_fix_prompt(raw_response, parse_error or "")

        response = llm_client.call(
            current_prompt,
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        raw_response = response.text

        template_output, parse_error = parse_template_output(raw_response)
        if template_output is not None:
            break

        logger.warning(
            "reauthor_template: attempt %d/%d failed to parse for canonical_id=%s: %s",
            attempt,
            max_attempts,
            canonical_id,
            parse_error,
        )

    if template_output is None:
        return ReauthorResult(
            status="rejected_invalid_json",
            canonical_problem_id=canonical_id,
            attempts=attempts,
            reasons=[f"LLM did not produce valid JSON after {attempts} attempt(s): {parse_error}"],
        )

    validation, variants = validate_template(template_output, canonical_record, variant_count=variant_count)

    if not validation.passed:
        return ReauthorResult(
            status="rejected_validation_failed",
            canonical_problem_id=canonical_id,
            attempts=attempts,
            template_output=template_output,
            validation=validation,
            reasons=validation.reasons,
        )

    provenance = Provenance(
        canonical_problem_id=canonical_id,
        source_list=list(canonical_record.get("source_list") or []),
        doocs_commit_hash=(canonical_record.get("ingest_trace") or {}).get("commit_hash"),
        license=(canonical_record.get("license_meta") or {}).get("license"),
        transform_model=llm_client.__class__.__name__,
        transform_seed=template_output.transform_seed or spec["seed"],
    )

    needs_review = bool(validation.reasons)
    template = Template(
        canonical_problem_id=canonical_id,
        pattern_id=None,  # set by the caller if a pattern_mining assignment lookup is available
        title=template_output.title,
        description=template_output.description,
        input_schema=template_output.input_schema,
        output_schema=template_output.output_schema,
        constraints=template_output.constraints,
        tie_breaker=template_output.tie_breaker,
        transform_seed=provenance.transform_seed,
        transform_model=provenance.transform_model,
        signature_hash=validation.signature_hash,
        provenance=provenance,
        needs_review=needs_review,
        review_reasons=list(validation.reasons),
    )

    for variant in variants:
        variant.template_id = template.template_id

    return ReauthorResult(
        status="accepted",
        canonical_problem_id=canonical_id,
        attempts=attempts,
        template_output=template_output,
        validation=validation,
        template=template,
        variants=variants,
    )
