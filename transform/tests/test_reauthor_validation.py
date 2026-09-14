"""End-to-end reauthor + validation tests: feed a canonical problem with
a real (executable) canonical Python solver, generate a template via a
mocked LLM, and assert the validation harness accepts a faithful rewrite
and rejects one that breaks algorithmic parity.
"""
import json

from transform.llm import MockLLMClient
from transform.models import TemplateOutput
from transform.reauthor import reauthor_template
from transform.validator import validate_template

from .fixtures import (
    GOOD_TWO_SUM_REWRITE,
    MERGE_INTERVALS_RECORD,
    NO_SOLUTION_RECORD,
    TWO_SUM_RECORD,
)


def test_reauthor_accepts_faithful_rewrite_end_to_end():
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))
    result = reauthor_template(TWO_SUM_RECORD, client, variant_count=5)

    assert result.status == "accepted"
    assert result.attempts == 1
    assert result.template is not None
    assert result.template.title == "Transaction Pair Match"
    assert result.template.needs_review is False
    assert result.template.signature_hash is not None
    assert result.template.provenance.canonical_problem_id == TWO_SUM_RECORD["id"]
    assert result.template.provenance.transform_model == "MockLLMClient"
    assert len(result.variants) == 5
    assert all(v.template_id == result.template.template_id for v in result.variants)


def test_reauthor_rejects_rewrite_that_breaks_sample_output():
    bad_rewrite = dict(GOOD_TWO_SUM_REWRITE)
    bad_rewrite["sample_public_tests"] = [{"input": "transactions=[2,7,11,15],target=9", "output": "[9,9]"}]

    client = MockLLMClient(default_response=json.dumps(bad_rewrite))
    result = reauthor_template(TWO_SUM_RECORD, client)

    assert result.status == "rejected_validation_failed"
    assert result.template is None
    assert any("sample_public_tests did not reproduce" in r for r in result.reasons)


def test_reauthor_rejects_rewrite_with_wrong_argument_count():
    bad_rewrite = dict(GOOD_TWO_SUM_REWRITE)
    bad_rewrite["sample_public_tests"] = [
        {"input": "transactions=[2,7,11,15],target=9,extra_field=1", "output": "[0,1]"}
    ]

    client = MockLLMClient(default_response=json.dumps(bad_rewrite))
    result = reauthor_template(TWO_SUM_RECORD, client)

    assert result.status == "rejected_validation_failed"


def test_reauthor_rejects_when_no_canonical_solution_available():
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))
    result = reauthor_template(NO_SOLUTION_RECORD, client)

    assert result.status == "rejected_validation_failed"
    assert any("no python canonical solution" in r for r in result.reasons)


def test_reauthor_flags_needs_review_when_tie_breaker_missing():
    rewrite = dict(GOOD_TWO_SUM_REWRITE)
    rewrite["tie_breaker"] = ""

    client = MockLLMClient(default_response=json.dumps(rewrite))
    result = reauthor_template(TWO_SUM_RECORD, client)

    assert result.status == "accepted"
    assert result.template.needs_review is True
    assert any("tie_breaker" in r for r in result.template.review_reasons)


def test_reauthor_retries_and_recovers_from_one_malformed_response():
    client = MockLLMClient(responses=["{{{not json", json.dumps(GOOD_TWO_SUM_REWRITE)])
    result = reauthor_template(TWO_SUM_RECORD, client, max_attempts=3)

    assert result.status == "accepted"
    assert result.attempts == 2
    assert len(client.calls) == 2


def test_reauthor_gives_up_after_max_attempts_exhausted():
    client = MockLLMClient(default_response="always broken")
    result = reauthor_template(TWO_SUM_RECORD, client, max_attempts=2)

    assert result.status == "rejected_invalid_json"
    assert result.attempts == 2
    assert len(client.calls) == 2


def test_validate_template_end_to_end_with_merge_intervals():
    rewrite = TemplateOutput(
        title="Consolidate Booking Windows",
        description="Merge overlapping booking windows.",
        input_schema={
            "type": "object",
            "properties": {"bookings": {"type": "array<array<integer>>"}},
        },
        output_schema={"type": "array<array<integer>>"},
        sample_public_tests=[
            {"input": "bookings=[[1,3],[2,6],[8,10],[15,18]]", "output": "[[1,6],[8,10],[15,18]]"}
        ],
        tie_breaker="not applicable",
        transform_seed="merge-seed",
    )

    result, variants = validate_template(rewrite, MERGE_INTERVALS_RECORD, variant_count=3)

    assert result.passed is True
    # nested array<array<integer>> is unsupported by the randomizer —
    # variant generation should be skipped, not silently produce garbage.
    assert variants == []
    assert any("skipped" in r for r in result.reasons)


def test_reauthor_uses_temperature_zero_by_default():
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))
    reauthor_template(TWO_SUM_RECORD, client)
    assert client.calls[0]["temperature"] == 0.0
