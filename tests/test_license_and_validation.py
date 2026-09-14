from ingestion.license_tracker import apply_license_policy, classify_license
from ingestion.models import CanonicalRecord, Difficulty, Example, LicenseMeta
from ingestion.validation import validate


def test_classify_license_permissive():
    review, public, internal = classify_license("MIT")
    assert review is False
    assert public is True
    assert internal is False


def test_classify_license_unknown_requires_review():
    review, public, internal = classify_license("unknown")
    assert review is True
    assert public is False
    assert internal is True


def test_classify_license_internal_only():
    review, public, internal = classify_license("educational")
    assert review is False
    assert public is False
    assert internal is True


def test_apply_license_policy_flags_record_for_review():
    record = CanonicalRecord(
        title="X", description="desc", license_meta=LicenseMeta(source="s", license="proprietary-secret", ingest_id="i1")
    )
    apply_license_policy(record)
    assert record.license_meta is not None
    assert record.license_meta.requires_legal_review is True
    assert record.validation.needs_review is True
    assert any("legal review" in r for r in record.validation.reasons)


def test_validate_flags_missing_fields():
    record = CanonicalRecord(title="", description="")
    validate(record)
    assert record.validation.needs_review is True
    assert any("missing title" in r for r in record.validation.reasons)
    assert any("missing description" in r for r in record.validation.reasons)


def test_validate_passes_clean_record():
    record = CanonicalRecord(
        title="Two Sum",
        description="Given an array of integers nums and an integer target, return indices of two numbers.",
        difficulty=Difficulty.EASY,
        tags=["array"],
        examples=[Example(input="[1,2]", output="[0,1]")],
        input_schema={"type": "array"},
        output_schema={"type": "array"},
        schema_inference_confidence=0.9,
    )
    validate(record)
    assert record.validation.needs_review is False
    assert record.validation.reasons == []
