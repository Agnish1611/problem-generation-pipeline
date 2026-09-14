"""Canonical data model for the Dataset Ingestion Layer.

Mirrors the schema described in the ingestion spec: a single normalized
record shape that every source adapter converges on, plus the supporting
value objects (examples, fingerprints, license metadata, ingest trace,
history/versioning).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    UNKNOWN = "unknown"


class Example(BaseModel):
    input: str
    output: str
    explanation: Optional[str] = None


class CanonicalSolution(BaseModel):
    languages: list[str] = Field(default_factory=list)
    code_hashes: list[str] = Field(default_factory=list)
    ast_signature: Optional[str] = None
    # The actual solution source text for the first language listed in
    # `languages`. Previously this was read off the adapter's RawProblem,
    # hashed for dedup/AST purposes, and then discarded — never reaching
    # unified_dataset.json/ingestion.db. Persisting it here is what makes
    # downstream validation (running the canonical solver against
    # generated variants) possible at all.
    code: Optional[str] = None


class Fingerprints(BaseModel):
    nl_embedding_id: Optional[str] = None
    code_embedding_id: Optional[str] = None
    content_hash: Optional[str] = None
    # Raw vectors are kept alongside ids so the similarity stages in this
    # repo can run without an external vector store. A real deployment
    # would push these into FAISS/Pinecone/Elastic and only keep the id.
    nl_embedding: Optional[list[float]] = None


class LicenseMeta(BaseModel):
    source: str
    source_url: Optional[str] = None
    license: str = "unknown"
    ingest_id: str
    requires_legal_review: bool = False
    public_allowed: bool = False
    internal_use_only: bool = True


class IngestTrace(BaseModel):
    parser: str
    raw_file: str
    raw_offset: Optional[int] = None
    commit_hash: Optional[str] = None


class HistoryEntry(BaseModel):
    step: str
    detail: str = ""
    at: datetime = Field(default_factory=_now)


class ValidationFlags(BaseModel):
    needs_review: bool = False
    needs_schema_review: bool = False
    requires_legal_review: bool = False
    reasons: list[str] = Field(default_factory=list)


class SchemaInference(BaseModel):
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class CanonicalRecord(BaseModel):
    """The unified record shape every adapter/normalizer output converges on."""

    id: str = Field(default_factory=_new_id)
    template_source_id: Optional[str] = None

    title: str
    description: str = ""
    difficulty: Difficulty = Difficulty.UNKNOWN
    tags: list[str] = Field(default_factory=list)

    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    schema_inference_confidence: float = 0.0

    examples: list[Example] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)

    has_solution: bool = False
    canonical_solution: Optional[CanonicalSolution] = None

    # True when the source flags this problem as paywalled/premium-only
    # (e.g. doocs/leetcode's "🔒" title suffix for LeetCode Premium
    # problems). Being premium does NOT by itself mean data is missing —
    # most premium problems still ship a full description and a
    # community solution. See `data_unavailable` for that.
    is_premium: bool = False
    # True only when the source's premium paywall actually left this
    # record without a usable solution (no adapter/source contributed
    # `solution_code`). This is what should gate "we're missing real
    # data for this problem" downstream, not `is_premium` alone.
    data_unavailable: bool = False

    fingerprints: Fingerprints = Field(default_factory=Fingerprints)
    license_meta: Optional[LicenseMeta] = None
    ingest_trace: Optional[IngestTrace] = None

    # De-duplication bookkeeping
    source_list: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    deduplicated_into: Optional[str] = None

    validation: ValidationFlags = Field(default_factory=ValidationFlags)

    version: int = 1
    history: list[HistoryEntry] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def touch(self, step: str, detail: str = "") -> None:
        self.history.append(HistoryEntry(step=step, detail=detail))
        self.updated_at = _now()

    def bump_version(self, step: str, detail: str = "") -> None:
        self.version += 1
        self.touch(step, detail)


class RawProblem(BaseModel):
    """Intermediate shape produced by source adapters before normalization.

    Adapters are deliberately permissive: they just lift raw fields into a
    common shape without cleaning or canonicalizing. That work happens in
    the Normalizer stage.
    """

    title: str
    description: str = ""
    difficulty: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    examples_raw: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    solution_code: Optional[str] = None
    solution_language: Optional[str] = None
    is_premium: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    # Provenance carried from the adapter
    source_name: str
    source_url: Optional[str] = None
    license: str = "unknown"
    raw_file: str
    raw_offset: Optional[int] = None
    parser: str
    commit_hash: Optional[str] = None
