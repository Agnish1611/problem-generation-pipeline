"""Data model for the re-authoring pipeline: the strict output shape an
LLM must produce for a rewritten template, plus the persisted `Template`
and `Variant` records that survive validation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class VariantGeneratorNotes(BaseModel):
    """LLM-declared guidance for generating additional test variants.
    Deliberately permissive (all optional) since this is model output,
    not something we control the shape of byte-for-byte.
    """

    model_config = ConfigDict(extra="ignore")

    n_range: Optional[list[Any]] = None       # advisory; model may return ints, floats, or strings
    value_range: Optional[list[Any]] = None   # advisory; model may return e.g. 'ASCII_Printable'
    special_flags: list[str] = Field(default_factory=list)


class TemplateOutput(BaseModel):
    """The strict JSON shape the LLM must return (OUTPUT_SCHEMA in the
    production prompt). Parsed straight from the model's response text;
    `extra="ignore"` so an LLM tacking on an extra field doesn't blow up
    parsing — required fields are still required, so a genuinely
    malformed response still fails validation and triggers a re-prompt.
    """

    model_config = ConfigDict(extra="ignore")

    title: str
    description: str
    input_format: str = ""
    output_format: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    tie_breaker: str = ""
    notes_for_variant_generator: VariantGeneratorNotes = Field(default_factory=VariantGeneratorNotes)
    sample_public_tests: list[dict[str, str]] = Field(default_factory=list)
    sample_hidden_tests_design: str = ""
    transform_seed: str = ""
    exemplar_core_logic: str = ""


class CaseCheck(BaseModel):
    """One executed check against the canonical solver — either one of
    the template's own `sample_public_tests` (positional-name mapped) or
    a generated randomized variant."""

    kind: str  # "sample_public_test" | "generated_variant"
    args: dict[str, Any]
    expected: Optional[Any] = None  # None for generated variants (no independent oracle)
    actual: Optional[Any] = None
    passed: bool = False
    error: Optional[str] = None
    timed_out: bool = False


class ValidationResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    cases: list[CaseCheck] = Field(default_factory=list)
    signature_hash: Optional[str] = None


class Provenance(BaseModel):
    canonical_problem_id: str
    source_list: list[str] = Field(default_factory=list)
    doocs_commit_hash: Optional[str] = None
    license: Optional[str] = None
    transform_model: str = ""
    transform_seed: str = ""


class Template(BaseModel):
    template_id: str = Field(default_factory=_new_id)
    canonical_problem_id: str
    pattern_id: Optional[str] = None
    title: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    tie_breaker: str = ""
    transform_seed: str = ""
    transform_model: str = ""
    signature_hash: Optional[str] = None
    provenance: Provenance
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Variant(BaseModel):
    variant_id: str = Field(default_factory=_new_id)
    template_id: str
    seed: str
    args: dict[str, Any]
    canonical_output: Any = None
    signature_hash: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class ReauthorResult(BaseModel):
    """Outcome of running the full reauthor control flow on one canonical
    record: build spec -> prompt LLM -> parse/repair JSON -> validate.
    """

    model_config = ConfigDict(extra="ignore")

    status: str  # "accepted" | "rejected_invalid_json" | "rejected_validation_failed"
    canonical_problem_id: str
    attempts: int = 0
    template_output: Optional[TemplateOutput] = None
    validation: Optional[ValidationResult] = None
    template: Optional[Template] = None
    variants: list[Variant] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
