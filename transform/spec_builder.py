"""Spec builder: converts a canonical record into a deterministic JSON
spec for the LLM, plus the exact system/user prompt strings to send.

Determinism matters here: the same canonical record + pattern + version
must always produce the same seed (and therefore, at temperature=0, the
same LLM output) so re-runs are reproducible and auditable.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

TRANSFORM_VERSION = "v1"

TRANSFORMATION_RULES = [
    "domain_swap",
    "add_tie_breaker",
    "avoid_direct_paraphrase",
    "preserve_algorithmic_core",
]

QUALITY_CHECKS = [
    "must preserve mapping from inputs -> outputs on canonical example(s)",
    "validate with canonical solver; signature must match",
]

SYSTEM_PROMPT = (
    'You are "Reauthor-Engine-v1", a structured problem rewriter. Your job is to '
    "rewrite an algorithmic canonical problem into a new, product-owned interview-style "
    "template that keeps the exact core algorithmic logic and input/output semantics but "
    "changes domain language, phrasing, and surface constraints to make the problem "
    "proprietary and not directly searchable. You must strictly follow the output JSON "
    'schema and nothing else. Do NOT include any extra commentary. If you cannot produce '
    'valid JSON, reply exactly with {"error":"<reason>"}.'
)

# Baked-in few-shot exemplars so prompts are stable across runs. Two
# examples covering different patterns (hash-map lookup, interval
# merging) per the "2-3 few-shot exemplars" guidance — keeps the prompt
# from overfitting to a single transformation style.
_FEW_SHOT_EXAMPLES = [
    {
        "canonical": {
            "title": "Two Sum",
            "description": (
                "Given an array of integers nums and a target, return indices of two "
                "numbers that add to target..."
            ),
            "input_schema": {"nums": "array<int>", "target": "int"},
            "output_schema": "array<int>",
            "examples": [{"input": "nums=[2,7,11,15],target=9", "output": "[0,1]"}],
            "tags": ["array", "hash-map"],
        },
        "output": {
            "title": "Transaction Pair Match",
            "description": (
                "You are given a list of transaction amounts in order of occurrence. Find "
                "the earliest pair of transactions whose amounts add up to a target "
                "reimbursement. Return the two timestamps (0-indexed) where the pair "
                "occurred. If multiple pairs are valid, return the pair with the smallest "
                "first timestamp; if still tied return the pair with smaller second "
                "timestamp."
            ),
            "input_format": "transactions: list of integers, target: integer",
            "output_format": "list of two indices [i,j]",
            "input_schema": {"transactions": "array<int>", "target": "int"},
            "output_schema": "array<int>",
            "constraints": [
                "2 <= transactions.length <= 10000",
                "-1e9 <= transactions[i] <= 1e9",
                "exactly one valid pair guaranteed in canonical tests",
            ],
            "tie_breaker": "prefer smallest first index, then smaller second index",
            "notes_for_variant_generator": {
                "n_range": [10, 10000],
                "value_range": [-1e9, 1e9],
                "special_flags": ["allow_negatives"],
            },
            "sample_public_tests": [{"input": "transactions=[2,7,11,15],target=9", "output": "[0,1]"}],
            "sample_hidden_tests_design": (
                "cases: duplicates, negative numbers, large n, pair at beginning, pair at "
                "end, repeated values"
            ),
            "transform_seed": "<seed>",
            "exemplar_core_logic": "Use a hash-map to store value->index and scan; for each x check target-x",
        },
    },
    {
        "canonical": {
            "title": "Merge Intervals",
            "description": (
                "Given an array of intervals where intervals[i] = [start_i, end_i], merge "
                "all overlapping intervals, and return an array of the non-overlapping "
                "intervals that cover all the intervals in the input."
            ),
            "input_schema": {"intervals": "array<array<int>>"},
            "output_schema": "array<array<int>>",
            "examples": [{"input": "intervals=[[1,3],[2,6],[8,10],[15,18]]", "output": "[[1,6],[8,10],[15,18]]"}],
            "tags": ["array", "sorting"],
        },
        "output": {
            "title": "Consolidate Booking Windows",
            "description": (
                "A venue receives a list of booking windows, each given as a [start, end] "
                "pair in hours since opening. Two windows that touch or overlap must be "
                "treated as a single continuous booking. Return the minimal list of "
                "non-overlapping windows that together cover every original booking, sorted "
                "by start time."
            ),
            "input_format": "bookings: list of [start, end] integer pairs",
            "output_format": "list of merged [start, end] integer pairs, sorted by start",
            "input_schema": {"bookings": "array<array<int>>"},
            "output_schema": "array<array<int>>",
            "constraints": [
                "1 <= bookings.length <= 10000",
                "0 <= start <= end <= 1e9",
                "windows touching at an endpoint (end_i == start_j) count as overlapping",
            ],
            "tie_breaker": "not applicable — merged result is uniquely determined",
            "notes_for_variant_generator": {
                "n_range": [1, 10000],
                "value_range": [0, 1e9],
                "special_flags": ["allow_touching_endpoints"],
            },
            "sample_public_tests": [
                {"input": "bookings=[[1,3],[2,6],[8,10],[15,18]]", "output": "[[1,6],[8,10],[15,18]]"}
            ],
            "sample_hidden_tests_design": (
                "cases: fully nested interval, single interval, all overlapping into one, "
                "no overlaps at all, touching endpoints"
            ),
            "transform_seed": "<seed>",
            "exemplar_core_logic": "Sort by start; sweep, merging into the last interval when the next start <= current end",
        },
    },
]

OUTPUT_SCHEMA_DESCRIPTION = {
    "title": "string (new product-owned title)",
    "description": "string (new problem text, <= 400 words)",
    "input_format": "string (human description)",
    "output_format": "string (human description)",
    "input_schema": "JSON Schema object",
    "output_schema": "JSON Schema object",
    "constraints": ["strings"],
    "tie_breaker": "string",
    "notes_for_variant_generator": {"n_range": [0, 0], "value_range": [0, 0], "special_flags": ["strings"]},
    "sample_public_tests": [{"input": "string", "output": "string"}],
    "sample_hidden_tests_design": "bullet list of cases (not exact data)",
    "transform_seed": "string (use the seed provided in input_spec)",
    "exemplar_core_logic": "1-2 short sentences describing required algorithmic idea",
}


def compute_transform_seed(canonical_problem_id: str, pattern: str, version: str = TRANSFORM_VERSION) -> str:
    """Deterministic seed: sha256(canonical_id + pattern + version). Used
    both to steer the LLM's domain-word/tie-breaker choices and, later,
    to derive per-variant seeds for the variant generator.
    """
    basis = f"{canonical_problem_id}:{pattern}:{version}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def infer_pattern_label(record: dict[str, Any], pattern_label: Optional[str] = None) -> str:
    """Best-effort short pattern label for the spec. Prefers an explicit
    label from pattern_mining's cluster assignment (passed in by the
    caller, since that requires a DB lookup this module shouldn't own);
    falls back to a heuristic built from the record's own tags.
    """
    if pattern_label:
        return pattern_label

    tags = [t for t in record.get("tags", []) if t and t != "unknown"]
    if tags:
        return "-".join(tags[:3])
    return "unclassified"


def build_spec(
    record: dict[str, Any],
    pattern_label: Optional[str] = None,
    version: str = TRANSFORM_VERSION,
) -> dict[str, Any]:
    """Builds the deterministic INPUT_SPEC dict for a canonical record.
    `record` is expected to be a canonical record dict (the shape found
    in unified_dataset.json).
    """
    canonical_id = record.get("id") or ""
    pattern = infer_pattern_label(record, pattern_label)
    seed = compute_transform_seed(canonical_id, pattern, version)

    examples = [
        {"input": ex.get("input", ""), "output": ex.get("output", "")} for ex in record.get("examples", [])
    ]

    return {
        "canonical_problem": {
            "id": canonical_id,
            "title": record.get("title", ""),
            "description": record.get("description", ""),
            "input_schema": record.get("input_schema", {}),
            "output_schema": record.get("output_schema", {}),
            "examples": examples,
            "tags": record.get("tags", []),
        },
        "pattern": pattern,
        "seed": seed,
        "transformation_rules": list(TRANSFORMATION_RULES),
        "quality_checks": list(QUALITY_CHECKS),
    }


def render_prompt(spec: dict[str, Any]) -> str:
    """Renders the exact production USER prompt (spec + few-shot examples
    + output schema + task instructions) as a single string, ready to
    pass as `prompt` to `LLMClient.call`.
    """
    parts: list[str] = []
    parts.append(
        "Below is a SPEC describing the canonical problem and the transformation rules. "
        "Produce a single JSON object strictly matching the OUTPUT_SCHEMA."
    )
    parts.append("")
    parts.append("INPUT_SPEC:")
    parts.append(json.dumps(spec, indent=2, ensure_ascii=False))
    parts.append("")
    parts.append("OUTPUT_SCHEMA (must be strict JSON, no markdown):")
    parts.append(json.dumps(OUTPUT_SCHEMA_DESCRIPTION, indent=2))
    parts.append("")
    parts.append("FEW-SHOT EXAMPLES: (for each example: canonical -> output)")
    for i, example in enumerate(_FEW_SHOT_EXAMPLES, start=1):
        parts.append("")
        parts.append(f"EXAMPLE {i}:")
        parts.append("Canonical:")
        parts.append(json.dumps(example["canonical"], indent=1, ensure_ascii=False))
        parts.append("Desired Output (JSON):")
        parts.append(json.dumps(example["output"], indent=1, ensure_ascii=False))
    parts.append("")
    parts.append(
        "(You must follow the structure of the JSON precisely and produce only JSON.)"
    )
    parts.append("")
    parts.append("TASK:")
    parts.append(
        "- Use the canonical_problem + pattern + seed + transformation_rules to output one "
        "JSON object that is a product-owned problem template."
    )
    parts.append("- Use the seed to deterministically pick domain words, tie-breakers, and sample test choices.")
    parts.append("- Keep temperature/creativity low: be literal, clear, and unambiguous.")
    parts.append("- Make description natural, concise, and interview-friendly.")
    parts.append("")
    parts.append("END.")
    return "\n".join(parts)


def build_json_fix_prompt(raw_response: str, error: str) -> str:
    """Prompt used to re-prompt the LLM when its previous response failed
    to parse as valid JSON matching TemplateOutput's required fields.
    """
    return (
        "Your previous response could not be parsed as valid JSON matching the required "
        "OUTPUT_SCHEMA. Reply again with ONLY a single valid JSON object, no markdown code "
        "fences, no commentary before or after.\n\n"
        f"Parse error: {error}\n\n"
        "Your previous response was:\n"
        f"{raw_response}\n\n"
        'If you truly cannot produce valid JSON, reply exactly with {"error":"<reason>"}.'
    )
