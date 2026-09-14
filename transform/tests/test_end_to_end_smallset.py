"""End-to-end test: run the template pipeline on a small set of canonical
records (mirroring the design doc's "run pipeline on 10 problems" smoke
test) and assert templates are produced and stored, incremental skip
works, and per-record failures don't take down the batch.
"""
import json

from transform.llm import MockLLMClient
from transform.pipeline import generate_templates_for_records
from transform.storage import read_template, read_templates_for_canonical_problem, read_variants_for_template

from .fixtures import (
    GOOD_TWO_SUM_REWRITE,
    MERGE_INTERVALS_RECORD,
    NO_SOLUTION_RECORD,
    TWO_SUM_RECORD,
)


def _ten_record_batch():
    """10 records: a mix of records that should accept, one that should
    be pre-filtered as data_unavailable, and duplicates of the same
    canonical problem to exercise the incremental skip path within a
    single batch isn't needed (skip is keyed by DB state, not batch
    position) — instead we vary ids so all 10 are genuinely distinct
    canonical problems for this test's purposes.
    """
    records = []
    for i in range(4):
        r = dict(TWO_SUM_RECORD)
        r["id"] = f"two-sum-clone-{i}"
        records.append(r)
    for i in range(4):
        r = dict(MERGE_INTERVALS_RECORD)
        r["id"] = f"merge-intervals-clone-{i}"
        records.append(r)
    for i in range(2):
        r = dict(NO_SOLUTION_RECORD)
        r["id"] = f"no-solution-clone-{i}"
        records.append(r)
    assert len(records) == 10
    return records


def _client_that_rewrites_by_title():
    merge_rewrite = {
        "title": "Consolidate Booking Windows",
        "description": "Merge overlapping booking windows.",
        "input_schema": {"type": "object", "properties": {"bookings": {"type": "array<array<integer>>"}}},
        "output_schema": {"type": "array<array<integer>>"},
        "sample_public_tests": [
            {"input": "bookings=[[1,3],[2,6],[8,10],[15,18]]", "output": "[[1,6],[8,10],[15,18]]"}
        ],
        "tie_breaker": "not applicable",
        "transform_seed": "merge-seed",
    }

    def responder(prompt: str, system):
        # The rendered prompt always embeds a "Merge Intervals" few-shot
        # exemplar (from spec_builder's baked-in examples) regardless of
        # which record is being rewritten, so a naive substring check
        # against the whole prompt would match every call. The actual
        # record under rewrite only appears in the INPUT_SPEC section
        # (between the "INPUT_SPEC:" and "OUTPUT_SCHEMA (must be strict"
        # markers), so restrict the check to that slice specifically —
        # not ".split('OUTPUT_SCHEMA')", since the intro sentence itself
        # contains the literal word "OUTPUT_SCHEMA" and would cut the
        # string short before INPUT_SPEC's content even starts.
        input_spec_section = prompt.split("INPUT_SPEC:")[1].split("OUTPUT_SCHEMA (must be strict")[0]
        if "Merge Intervals" in input_spec_section:
            return json.dumps(merge_rewrite)
        return json.dumps(GOOD_TWO_SUM_REWRITE)

    return MockLLMClient(response_fn=responder)


def test_pipeline_on_ten_record_batch_produces_expected_outcomes(tmp_path):
    db_path = tmp_path / "test.db"
    records = _ten_record_batch()
    client = _client_that_rewrites_by_title()

    metrics = generate_templates_for_records(records, db_path=db_path, llm_client=client, variant_count=3)

    assert metrics.records_considered == 10
    assert metrics.accepted == 8  # 4 two-sum clones + 4 merge-intervals clones
    assert metrics.rejected_validation_failed == 2  # the 2 no-solution clones, pre-filtered
    assert metrics.rejected_invalid_json == 0
    assert metrics.records_skipped_existing == 0
    assert metrics.total_variants_generated > 0
    assert 0.0 < metrics.acceptance_rate() < 1.0

    # Spot-check one accepted template actually persisted correctly.
    templates = read_templates_for_canonical_problem("two-sum-clone-0", db_path)
    assert len(templates) == 1
    assert templates[0]["title"] == "Transaction Pair Match"

    variants = read_variants_for_template(templates[0]["template_id"], db_path)
    assert len(variants) == 3


def test_pipeline_is_incremental_by_default(tmp_path):
    db_path = tmp_path / "test.db"
    records = [dict(TWO_SUM_RECORD, id="incremental-test-001")]
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))

    first_run = generate_templates_for_records(records, db_path=db_path, llm_client=client)
    assert first_run.accepted == 1
    assert first_run.records_skipped_existing == 0

    second_run = generate_templates_for_records(records, db_path=db_path, llm_client=client)
    assert second_run.accepted == 0
    assert second_run.records_skipped_existing == 1
    # The LLM must not be called again for a record that's already templated.
    assert len(client.calls) == 1


def test_pipeline_force_reruns_already_templated_records(tmp_path):
    db_path = tmp_path / "test.db"
    records = [dict(TWO_SUM_RECORD, id="force-test-001")]
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))

    generate_templates_for_records(records, db_path=db_path, llm_client=client)
    second_run = generate_templates_for_records(records, db_path=db_path, llm_client=client, force=True)

    assert second_run.accepted == 1
    assert second_run.records_skipped_existing == 0
    assert len(client.calls) == 2


def test_pipeline_continues_after_a_rejected_record(tmp_path):
    db_path = tmp_path / "test.db"
    records = [
        dict(NO_SOLUTION_RECORD, id="reject-me"),
        dict(TWO_SUM_RECORD, id="accept-me"),
    ]
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))

    metrics = generate_templates_for_records(records, db_path=db_path, llm_client=client)

    assert metrics.records_considered == 2
    assert metrics.accepted == 1
    assert metrics.rejected_validation_failed == 1
    assert read_templates_for_canonical_problem("accept-me", db_path)
    assert not read_templates_for_canonical_problem("reject-me", db_path)


def test_pipeline_records_llm_call_error_without_crashing_batch(tmp_path):
    from transform.llm import LLMCallError

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def call(self, *args, **kwargs):
            self.calls += 1
            raise LLMCallError("simulated connection failure")

    db_path = tmp_path / "test.db"
    records = [dict(TWO_SUM_RECORD, id="flaky-test")]
    metrics = generate_templates_for_records(records, db_path=db_path, llm_client=FlakyClient())

    assert metrics.llm_errors == 1
    assert metrics.accepted == 0
