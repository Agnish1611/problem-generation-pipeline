"""Tests for parsing/repairing LLM JSON responses (reauthor.py's
extract_json_block / parse_template_output), using mocked LLM responses
— good JSON, slightly malformed JSON, markdown-fenced JSON, and the
model's own declared-error shape.
"""
import json

from transform.reauthor import extract_json_block, parse_template_output

from .fixtures import GOOD_TWO_SUM_REWRITE


def test_parse_template_output_accepts_well_formed_json():
    raw = json.dumps(GOOD_TWO_SUM_REWRITE)
    output, error = parse_template_output(raw)
    assert error is None
    assert output is not None
    assert output.title == "Transaction Pair Match"
    assert output.tie_breaker == "prefer smallest first index"


def test_parse_template_output_rejects_missing_required_fields():
    # "title" and "description" are required on TemplateOutput.
    raw = json.dumps({"tie_breaker": "x"})
    output, error = parse_template_output(raw)
    assert output is None
    assert error is not None
    assert "schema validation failed" in error


def test_parse_template_output_rejects_non_json_text():
    output, error = parse_template_output("this is not json at all")
    assert output is None
    assert "JSONDecodeError" in error


def test_parse_template_output_rejects_json_array_instead_of_object():
    output, error = parse_template_output("[1, 2, 3]")
    assert output is None
    assert "expected a JSON object" in error


def test_parse_template_output_handles_model_declared_error():
    raw = json.dumps({"error": "cannot rewrite this problem safely"})
    output, error = parse_template_output(raw)
    assert output is None
    assert "cannot rewrite this problem safely" in error


def test_parse_template_output_ignores_unknown_extra_fields():
    payload = dict(GOOD_TWO_SUM_REWRITE)
    payload["some_field_the_model_made_up"] = "surprise"
    output, error = parse_template_output(json.dumps(payload))
    assert error is None
    assert output.title == "Transaction Pair Match"


def test_extract_json_block_strips_markdown_fences():
    fenced = "```json\n" + json.dumps(GOOD_TWO_SUM_REWRITE) + "\n```"
    extracted = extract_json_block(fenced)
    parsed = json.loads(extracted)
    assert parsed["title"] == "Transaction Pair Match"


def test_extract_json_block_extracts_object_from_surrounding_prose():
    prose = "Sure, here's the JSON you asked for:\n" + json.dumps({"title": "X", "description": "Y"}) + "\nHope that helps!"
    extracted = extract_json_block(prose)
    parsed = json.loads(extracted)
    assert parsed == {"title": "X", "description": "Y"}


def test_extract_json_block_returns_original_text_if_no_braces_found():
    text = "no json here at all"
    assert extract_json_block(text) == text
