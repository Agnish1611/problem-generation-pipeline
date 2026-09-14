from transform.spec_builder import (
    build_json_fix_prompt,
    build_spec,
    compute_transform_seed,
    infer_pattern_label,
    render_prompt,
)

from .fixtures import TWO_SUM_RECORD


def test_compute_transform_seed_is_deterministic():
    a = compute_transform_seed("id-1", "pair-sum")
    b = compute_transform_seed("id-1", "pair-sum")
    assert a == b


def test_compute_transform_seed_differs_for_different_inputs():
    a = compute_transform_seed("id-1", "pair-sum")
    b = compute_transform_seed("id-2", "pair-sum")
    c = compute_transform_seed("id-1", "sliding-window")
    assert a != b
    assert a != c


def test_infer_pattern_label_prefers_explicit_label():
    label = infer_pattern_label(TWO_SUM_RECORD, pattern_label="hash-map-lookup")
    assert label == "hash-map-lookup"


def test_infer_pattern_label_falls_back_to_tags():
    label = infer_pattern_label(TWO_SUM_RECORD)
    assert "array" in label


def test_infer_pattern_label_falls_back_to_unclassified_with_no_tags():
    label = infer_pattern_label({"tags": []})
    assert label == "unclassified"


def test_build_spec_roundtrip_is_deterministic():
    spec1 = build_spec(TWO_SUM_RECORD)
    spec2 = build_spec(TWO_SUM_RECORD)
    assert spec1 == spec2
    assert spec1["seed"] == spec2["seed"]


def test_build_spec_contains_expected_shape():
    spec = build_spec(TWO_SUM_RECORD, pattern_label="pair-sum")
    assert spec["canonical_problem"]["id"] == TWO_SUM_RECORD["id"]
    assert spec["canonical_problem"]["title"] == "Two Sum"
    assert spec["pattern"] == "pair-sum"
    assert len(spec["canonical_problem"]["examples"]) == 2
    assert "domain_swap" in spec["transformation_rules"]
    assert "seed" in spec


def test_build_spec_seed_changes_with_pattern_label():
    spec_a = build_spec(TWO_SUM_RECORD, pattern_label="pattern-a")
    spec_b = build_spec(TWO_SUM_RECORD, pattern_label="pattern-b")
    assert spec_a["seed"] != spec_b["seed"]


def test_render_prompt_includes_spec_and_schema_and_examples():
    spec = build_spec(TWO_SUM_RECORD)
    prompt = render_prompt(spec)
    assert "INPUT_SPEC:" in prompt
    assert "OUTPUT_SCHEMA" in prompt
    assert "FEW-SHOT EXAMPLES" in prompt
    assert "Two Sum" in prompt
    assert spec["seed"] in prompt


def test_render_prompt_is_deterministic_for_same_spec():
    spec = build_spec(TWO_SUM_RECORD)
    assert render_prompt(spec) == render_prompt(spec)


def test_build_json_fix_prompt_includes_raw_response_and_error():
    prompt = build_json_fix_prompt("not valid json", "JSONDecodeError: bad token")
    assert "not valid json" in prompt
    assert "JSONDecodeError" in prompt
    # Fix prompt is short and references the schema by name but does not
    # repeat the full OUTPUT_SCHEMA JSON dump (unlike the main prompt).
    assert "exemplar_core_logic" not in prompt
