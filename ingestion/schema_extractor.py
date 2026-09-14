"""Schema Extractor.

Heuristically infers a minimal JSON Schema for a problem's input/output
shape from its structured examples. This is deliberately conservative: if
we can't confidently infer a shape we set a low confidence score and let
the validation gate flag the record for `needs_schema_review` rather than
guessing and shipping a wrong schema downstream.
"""
from __future__ import annotations

import json
import re

from .models import CanonicalRecord, Example, SchemaInference

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def _infer_json_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        if value:
            return f"array<{_infer_json_type(value[0])}>"
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _try_parse_scalar(token: str):
    token = token.strip()
    if _INT_RE.match(token):
        try:
            return int(token)
        except ValueError:
            return token
    if _FLOAT_RE.match(token):
        try:
            return float(token)
        except ValueError:
            return token
    if token.lower() in ("true", "false"):
        return token.lower() == "true"
    return token


def _parse_example_side(text: str):
    """Attempts to parse an example's input/output text into structured
    values. Handles JSON directly; falls back to splitting on commas for
    LeetCode-style `nums = [1,2,3], target = 9` inputs; falls back to a
    single raw scalar/string otherwise.
    """
    text = text.strip()
    if not text:
        return None, 0.0

    # Try strict JSON first (highest confidence).
    try:
        return json.loads(text), 1.0
    except (json.JSONDecodeError, ValueError):
        pass

    # LeetCode-style "name = value, name2 = value2"
    if "=" in text:
        parts = [p.strip() for p in re.split(r",(?![^\[]*\])", text) if "=" in p]
        if parts:
            result = {}
            ok = True
            for part in parts:
                name, _, val = part.partition("=")
                name = name.strip()
                val = val.strip()
                try:
                    parsed = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    parsed = _try_parse_scalar(val)
                if not name:
                    ok = False
                    break
                result[name] = parsed
            if ok and result:
                return result, 0.85

    # Bare scalar
    return _try_parse_scalar(text), 0.5


def _schema_from_values(values: list) -> tuple[dict, float]:
    """Builds a JSON-schema-ish dict describing the shape shared across a
    list of parsed example values, plus a confidence score."""
    if not values:
        return {}, 0.0

    if all(isinstance(v, dict) for v in values):
        keys: dict[str, list] = {}
        for v in values:
            for k, val in v.items():
                keys.setdefault(k, []).append(val)
        properties = {k: {"type": _infer_json_type(vals[0])} for k, vals in keys.items()}
        # Confidence drops if keys aren't consistent across examples.
        key_sets = [frozenset(v.keys()) for v in values]
        consistent = len(set(key_sets)) == 1
        return (
            {"type": "object", "properties": properties, "required": list(properties)},
            0.9 if consistent else 0.6,
        )

    types = {_infer_json_type(v) for v in values}
    if len(types) == 1:
        return {"type": next(iter(types))}, 0.8
    return {"type": "any"}, 0.3


def infer_schema(examples: list[Example]) -> SchemaInference:
    if not examples:
        return SchemaInference(input_schema={}, output_schema={}, confidence=0.0)

    input_values = []
    output_values = []
    parse_confidences = []

    for ex in examples:
        in_val, in_conf = _parse_example_side(ex.input)
        out_val, out_conf = _parse_example_side(ex.output)
        input_values.append(in_val)
        output_values.append(out_val)
        parse_confidences.append(min(in_conf, out_conf))

    input_schema, in_shape_conf = _schema_from_values(input_values)
    output_schema, out_shape_conf = _schema_from_values(output_values)

    avg_parse_conf = sum(parse_confidences) / len(parse_confidences)
    overall = round(avg_parse_conf * 0.5 + ((in_shape_conf + out_shape_conf) / 2) * 0.5, 3)

    return SchemaInference(
        input_schema=input_schema,
        output_schema=output_schema,
        confidence=overall,
    )


def apply_schema_inference(record: CanonicalRecord) -> CanonicalRecord:
    inference = infer_schema(record.examples)
    record.input_schema = inference.input_schema
    record.output_schema = inference.output_schema
    record.schema_inference_confidence = inference.confidence
    record.touch("schema_inferred", detail=f"confidence={inference.confidence}")
    return record
