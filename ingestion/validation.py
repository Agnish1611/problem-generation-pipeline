"""Validation & Sanity Checks (spec section 5).

Light-weight gate checks run at ingestion time. Failing a gate does not
drop the record — it flags it for human/automated review downstream,
consistent with "fail early and store failure reasons for human review".
"""
from __future__ import annotations

from .models import CanonicalRecord, Difficulty

MIN_DESCRIPTION_LENGTH = 20
MAX_DESCRIPTION_LENGTH = 20_000
SCHEMA_CONFIDENCE_THRESHOLD = 0.5


def validate(record: CanonicalRecord) -> CanonicalRecord:
    reasons: list[str] = list(record.validation.reasons)  # preserve license reasons etc.
    needs_review = record.validation.needs_review
    needs_schema_review = False

    if not record.title.strip():
        reasons.append("missing title")
        needs_review = True

    if not record.description.strip():
        reasons.append("missing description")
        needs_review = True
    elif len(record.description) < MIN_DESCRIPTION_LENGTH:
        reasons.append(f"description shorter than {MIN_DESCRIPTION_LENGTH} chars")
        needs_review = True
    elif len(record.description) > MAX_DESCRIPTION_LENGTH:
        reasons.append(f"description longer than {MAX_DESCRIPTION_LENGTH} chars")
        needs_review = True

    if record.difficulty == Difficulty.UNKNOWN:
        reasons.append("difficulty unknown/unnormalized")
        needs_review = True

    if not record.tags or record.tags == ["unknown"]:
        reasons.append("no tags inferred")
        needs_review = True

    if record.data_unavailable:
        reasons.append("premium-only problem with no solution/data available from any source")
        needs_review = True

    if record.examples:
        if record.schema_inference_confidence < SCHEMA_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"schema_inference_confidence {record.schema_inference_confidence} below threshold {SCHEMA_CONFIDENCE_THRESHOLD}"
            )
            needs_schema_review = True
            needs_review = True
    else:
        reasons.append("no examples to infer schema from")
        needs_schema_review = True
        needs_review = True

    record.validation.reasons = reasons
    record.validation.needs_review = needs_review
    record.validation.needs_schema_review = needs_schema_review

    record.touch("validated", detail=f"needs_review={needs_review}")
    return record


def validate_batch(records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    return [validate(r) for r in records]
