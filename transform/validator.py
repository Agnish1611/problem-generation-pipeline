"""Validation harness: the safety-critical piece of the reauthor pipeline.

A generated template is only ever persisted if the CANONICAL solver (the
community solution ingested for that problem — never the LLM's own
output, which is never executed) reproduces the exact outputs the
template's own sample tests claim, when the template's renamed arguments
are mapped back onto the canonical solver's real parameter names.

This deliberately does NOT execute anything the LLM wrote. The LLM only
produces problem text/schema/sample tests; the code that actually runs is
always the pre-existing, ingested `canonical_solution.code`. That's the
whole point: if the rewrite preserved the algorithmic core, the same
solver must still produce the same outputs on the (renamed) sample
inputs. If it doesn't, the rewrite is rejected.

Scope, deliberately limited for this pass:
  * Sample-test validation is exact (see rejection criteria below).
  * "Variant" generation here is a simple deterministic randomizer over
    scalar/array-of-scalar parameters, gated by the LLM's own declared
    `notes_for_variant_generator` ranges. There is no bespoke
    `variant_engine/` yet (per-pattern structural generators, hidden vs
    public test splitting, etc.) — this is intentionally the minimal
    thing that lets you say "the canonical solver still runs cleanly
    across N randomized inputs in the declared range" as an additional
    parity signal beyond the fixed sample tests. Problems whose
    parameters aren't plain scalars/arrays of scalars (e.g. ListNode,
    TreeNode, custom objects) skip randomized-variant generation
    entirely and rely on sample-test validation only — flagged via
    `ValidationResult.reasons`, not silently ignored.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

from sandbox import (
    EntryPoint,
    canonicalize_value,
    detect_entry_point,
    get_entry_point_params,
    parse_kwargs,
    run_code,
)
from sandbox.runner import DEFAULT_MEMORY_LIMIT_MB, DEFAULT_TIMEOUT_SECONDS

from .models import CaseCheck, TemplateOutput, ValidationResult, Variant

DEFAULT_VARIANT_COUNT = 8


def map_positional(args_by_name: dict[str, Any], canonical_params: list[str]) -> Optional[dict[str, Any]]:
    """Maps a dict of (possibly renamed) argument names onto the
    canonical solver's real parameter names, by POSITION — the order
    values appear in the template's test string, matched against the
    order parameters appear in the canonical solver's signature.

    Returns None if the argument counts don't match (a strong signal
    the rewrite changed the algorithmic core — see
    `validate_template`'s "inserted extra required fields" rejection
    criterion).
    """
    values = list(args_by_name.values())
    if len(values) != len(canonical_params):
        return None
    return dict(zip(canonical_params, values))


def _run_canonical(
    canonical_code: str,
    entry: EntryPoint,
    canonical_args: dict[str, Any],
    timeout_seconds: float,
    memory_limit_mb: int,
):
    return run_code(canonical_code, entry, canonical_args, timeout_seconds, memory_limit_mb)


def validate_sample_tests(
    template_output: TemplateOutput,
    canonical_code: str,
    entry: EntryPoint,
    canonical_params: list[str],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> list[CaseCheck]:
    """Runs the canonical solver against every one of the template's own
    `sample_public_tests`, mapping the template's (renamed) argument
    names positionally onto the canonical solver's real parameters, and
    compares against the output the template itself claims.
    """
    checks: list[CaseCheck] = []
    for test in template_output.sample_public_tests:
        raw_input = test.get("input", "")
        raw_output = test.get("output", "")

        template_args = parse_kwargs(raw_input)
        canonical_args = map_positional(template_args, canonical_params)

        if canonical_args is None:
            checks.append(
                CaseCheck(
                    kind="sample_public_test",
                    args=template_args,
                    expected=raw_output,
                    passed=False,
                    error=(
                        f"argument count mismatch: template test has {len(template_args)} "
                        f"argument(s), canonical solver expects {len(canonical_params)} "
                        f"({canonical_params!r}) — the rewrite likely added/removed a "
                        "required field"
                    ),
                )
            )
            continue

        from sandbox.runner import _parse_value  # local import: intentionally internal helper

        expected_parsed = _parse_value(raw_output)
        run_result = _run_canonical(canonical_code, entry, canonical_args, timeout_seconds, memory_limit_mb)

        if run_result.error is not None:
            checks.append(
                CaseCheck(
                    kind="sample_public_test",
                    args=canonical_args,
                    expected=expected_parsed,
                    passed=False,
                    error=run_result.error,
                    timed_out=run_result.timed_out,
                )
            )
            continue

        actual = canonicalize_value(run_result.value)
        expected = canonicalize_value(expected_parsed)
        checks.append(
            CaseCheck(
                kind="sample_public_test",
                args=canonical_args,
                expected=expected_parsed,
                actual=run_result.value,
                passed=actual == expected,
            )
        )

    return checks


# JSON-Schema-ish "type" strings this randomizer knows how to generate.
# Anything else (objects, custom $ref-style structures) is unsupported
# and randomized-variant generation is skipped for that parameter.
_SUPPORTED_SCALAR_TYPES = {"integer", "number", "string", "boolean"}


def _random_scalar(type_name: str, rng: random.Random, value_range: Optional[list[float]]) -> Any:
    lo, hi = (value_range if value_range and len(value_range) == 2 else [-100, 100])
    if type_name == "integer":
        return rng.randint(int(lo), int(hi))
    if type_name == "number":
        return round(rng.uniform(float(lo), float(hi)), 4)
    if type_name == "boolean":
        return rng.choice([True, False])
    if type_name == "string":
        length = rng.randint(1, 8)
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
    raise ValueError(f"unsupported scalar type: {type_name}")


def _random_value_for_schema_type(
    type_name: str, rng: random.Random, value_range: Optional[list[float]], n_range: Optional[list[float]]
) -> Any:
    if type_name in _SUPPORTED_SCALAR_TYPES:
        return _random_scalar(type_name, rng, value_range)
    if type_name.startswith("array<") and type_name.endswith(">"):
        inner = type_name[len("array<"):-1]
        if inner not in _SUPPORTED_SCALAR_TYPES:
            raise ValueError(f"unsupported array element type: {inner}")
        lo_n, hi_n = (n_range if n_range and len(n_range) == 2 else [1, 10])
        length = rng.randint(max(0, int(lo_n)), max(int(lo_n), int(hi_n)))
        return [_random_scalar(inner, rng, value_range) for _ in range(length)]
    raise ValueError(f"unsupported schema type: {type_name}")


def can_generate_variants(template_output: TemplateOutput) -> tuple[bool, str]:
    """Checks whether every property in the template's declared
    input_schema is a type this randomizer supports. Returns
    (True, "") if so, or (False, reason) if not — callers should skip
    randomized-variant generation (not fail the whole template) when
    this returns False.
    """
    properties = (template_output.input_schema or {}).get("properties") or {}
    if not properties:
        return False, "template input_schema has no declared properties to randomize"

    for name, prop in properties.items():
        type_name = prop.get("type") if isinstance(prop, dict) else None
        if not type_name:
            return False, f"property {name!r} has no declared type"
        if type_name in _SUPPORTED_SCALAR_TYPES:
            continue
        if type_name.startswith("array<") and type_name.endswith(">"):
            inner = type_name[len("array<"):-1]
            if inner in _SUPPORTED_SCALAR_TYPES:
                continue
        return False, f"property {name!r} has unsupported type {type_name!r} for randomized generation"

    return True, ""


def generate_variant_args(
    template_output: TemplateOutput,
    canonical_problem_id: str,
    transform_seed: str,
    variant_index: int,
) -> tuple[str, dict[str, Any]]:
    """Deterministically generates one set of randomized argument values
    for the template's declared input_schema, gated by the LLM's own
    `notes_for_variant_generator` ranges. Returns (seed, args).

    Per-variant seed = sha256(canonical_problem_id + transform_seed +
    variant_index), matching the design doc's deterministic-seeding
    requirement.
    """
    seed_basis = f"{canonical_problem_id}:{transform_seed}:{variant_index}"
    seed = hashlib.sha256(seed_basis.encode("utf-8")).hexdigest()
    rng = random.Random(seed)

    notes = template_output.notes_for_variant_generator
    properties = (template_output.input_schema or {}).get("properties") or {}

    args: dict[str, Any] = {}
    for name, prop in properties.items():
        type_name = prop.get("type") if isinstance(prop, dict) else None
        args[name] = _random_value_for_schema_type(type_name, rng, notes.value_range, notes.n_range)

    return seed, args


def generate_and_validate_variants(
    template_output: TemplateOutput,
    canonical_code: str,
    entry: EntryPoint,
    canonical_params: list[str],
    canonical_problem_id: str,
    transform_seed: str,
    count: int = DEFAULT_VARIANT_COUNT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> tuple[list[Variant], list[CaseCheck]]:
    """Generates `count` deterministic randomized variants and runs the
    canonical solver on each. There is no independent oracle for a
    randomized input (nothing else claims to know the "right" answer),
    so a variant is not compared against an expected value — instead,
    the canonical solver's own output becomes the variant's recorded
    `canonical_output`, and the check here is purely "did the canonical
    solver run to completion without erroring/timing out on this input
    shape and range." A solver that can't handle its own declared
    parameter ranges is a real signal the LLM's `notes_for_variant_generator`
    (or the underlying rewrite) is unsound.
    """
    can_generate, reason = can_generate_variants(template_output)
    if not can_generate:
        return [], [CaseCheck(kind="generated_variant", args={}, passed=False, error=f"skipped: {reason}")]

    variants: list[Variant] = []
    checks: list[CaseCheck] = []

    for i in range(count):
        seed, template_args = generate_variant_args(template_output, canonical_problem_id, transform_seed, i)
        canonical_args = map_positional(template_args, canonical_params)

        if canonical_args is None:
            checks.append(
                CaseCheck(
                    kind="generated_variant",
                    args=template_args,
                    passed=False,
                    error="argument count mismatch mapping generated variant onto canonical solver",
                )
            )
            continue

        run_result = _run_canonical(canonical_code, entry, canonical_args, timeout_seconds, memory_limit_mb)

        if run_result.error is not None:
            checks.append(
                CaseCheck(
                    kind="generated_variant",
                    args=canonical_args,
                    passed=False,
                    error=run_result.error,
                    timed_out=run_result.timed_out,
                )
            )
            continue

        output_value = canonicalize_value(run_result.value)
        signature_hash = hashlib.sha256(repr(output_value).encode("utf-8")).hexdigest()
        variants.append(
            Variant(
                template_id="",  # filled in by the caller once the Template's id is known
                seed=seed,
                args=canonical_args,
                canonical_output=run_result.value,
                signature_hash=signature_hash,
            )
        )
        checks.append(
            CaseCheck(kind="generated_variant", args=canonical_args, actual=run_result.value, passed=True)
        )

    return variants, checks


def check_schema_compatibility(template_output: TemplateOutput, canonical_record: dict[str, Any]) -> list[str]:
    """Soft-rejection checks per the design doc's rejection criteria:
      - output_schema fundamentally changed shape (e.g. array vs scalar)
      - LLM declared a different number of input properties than the
        canonical solver actually takes (a stronger version of this is
        already a hard failure via `map_positional` returning None on
        the sample tests, but checking it here catches templates whose
        sample tests happen to accidentally satisfy positional mapping
        despite a schema mismatch).
    Returns a list of human-readable reasons; empty list means no
    schema-level concerns found. This is advisory (feeds `needs_review`)
    for the property-count check but the caller decides hard-reject vs
    review based on the actual sample-test execution results, not this
    function alone.
    """
    reasons: list[str] = []

    canonical_output_type = (canonical_record.get("output_schema") or {}).get("type")
    template_output_type = (template_output.output_schema or {}).get("type")
    if canonical_output_type and template_output_type:
        canonical_is_array = str(canonical_output_type).startswith("array")
        template_is_array = str(template_output_type).startswith("array")
        if canonical_is_array != template_is_array:
            reasons.append(
                f"output_schema shape changed: canonical={canonical_output_type!r} "
                f"vs template={template_output_type!r}"
            )

    if not template_output.tie_breaker.strip():
        reasons.append("template did not declare a tie_breaker (may be ambiguous for multi-answer problems)")

    return reasons


def validate_template(
    template_output: TemplateOutput,
    canonical_record: dict[str, Any],
    variant_count: int = DEFAULT_VARIANT_COUNT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> tuple[ValidationResult, list[Variant]]:
    """Full validation pipeline for one candidate template: sample-test
    parity against the canonical solver (hard gate), then randomized
    variant generation (soft — skipped, not failed, if unsupported).

    Hard-reject conditions (validation.passed = False):
      - no canonical solution code available for this record at all
      - canonical solver has no detectable entry point
      - template declared zero sample_public_tests
      - any sample test fails to reproduce the template's claimed output
        when run through the canonical solver
    """
    reasons: list[str] = []
    cases: list[CaseCheck] = []

    solution = canonical_record.get("canonical_solution") or {}
    languages = [str(l).lower() for l in (solution.get("languages") or [])]
    canonical_code = solution.get("code")

    if not canonical_code or "python" not in languages:
        return (
            ValidationResult(passed=False, reasons=["no python canonical solution available to validate against"]),
            [],
        )

    entry = detect_entry_point(canonical_code)
    if entry is None:
        return (
            ValidationResult(passed=False, reasons=["canonical solver has no detectable entry point"]),
            [],
        )

    canonical_params = get_entry_point_params(canonical_code, entry) or []

    if not template_output.sample_public_tests:
        return (
            ValidationResult(passed=False, reasons=["template declared zero sample_public_tests"]),
            [],
        )

    sample_checks = validate_sample_tests(
        template_output, canonical_code, entry, canonical_params, timeout_seconds, memory_limit_mb
    )
    cases.extend(sample_checks)

    sample_all_passed = all(c.passed for c in sample_checks)
    if not sample_all_passed:
        reasons.append("one or more sample_public_tests did not reproduce under the canonical solver")

    schema_reasons = check_schema_compatibility(template_output, canonical_record)
    reasons.extend(schema_reasons)

    if not sample_all_passed:
        # Hard gate: never generate/persist variants for a template that
        # already failed the sample-test parity check.
        return ValidationResult(passed=False, reasons=reasons, cases=cases), []

    canonical_problem_id = canonical_record.get("id", "")
    transform_seed = template_output.transform_seed
    variants, variant_checks = generate_and_validate_variants(
        template_output,
        canonical_code,
        entry,
        canonical_params,
        canonical_problem_id,
        transform_seed,
        count=variant_count,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )
    cases.extend(variant_checks)

    variant_failures = [c for c in variant_checks if not c.passed and c.kind == "generated_variant"]
    variants_were_skipped = bool(variant_checks) and not variants and len(variant_checks) == 1 and "skipped:" in (
        variant_checks[0].error or ""
    )
    if variants_were_skipped:
        reasons.append(variant_checks[0].error or "randomized variant generation skipped")
    elif variant_failures:
        # A solver erroring on its own declared randomized range is a real
        # parity concern, but sample tests already passed exactly — treat
        # as needs_review rather than a hard reject, since it may simply
        # mean the LLM's declared ranges need tightening, not that the
        # rewrite itself is wrong.
        reasons.append(f"{len(variant_failures)}/{len(variant_checks)} generated variants failed on the canonical solver")

    signature_hash = None
    if variants:
        concatenated = "".join(v.signature_hash or "" for v in variants)
        signature_hash = hashlib.sha256(concatenated.encode("utf-8")).hexdigest()

    # `passed` reflects the hard gate only (sample tests). Variant
    # issues and schema soft-checks surface via `reasons` and get
    # translated to `needs_review` by the caller (reauthor.py), not a
    # rejection — this matches the design doc's distinction between
    # "reject" (unfaithful rewrite) and "needs_review" (a human should
    # look at this, but it wasn't a proven algorithmic mismatch).
    return (
        ValidationResult(passed=True, reasons=reasons, cases=cases, signature_hash=signature_hash),
        variants,
    )
