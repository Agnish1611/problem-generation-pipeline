"""Pipeline Orchestrator.

Wires together: Source Adapter -> Normalizer -> Schema Extractor ->
Fingerprinting -> License/Provenance -> De-duplication -> Validation ->
Storage. Mirrors the flow in spec section 3.

Idempotency: each source is keyed by its file path (`source_key`). The
adapters emit `raw_offset` per record, and `Storage.upsert` matches on
`(source_name, raw_file, raw_offset)` so re-running the same file updates
existing records instead of duplicating them (spec section 9).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .adapters import SourceAdapter
from .dedup import consolidate
from .fingerprint import apply_fingerprints
from .license_tracker import apply_license_policy
from .models import CanonicalRecord
from .normalizer import normalize
from .schema_extractor import apply_schema_inference
from .storage import Storage, export_unified_dataset
from .validation import validate_batch

logger = logging.getLogger(__name__)


@dataclass
class IngestSource:
    """One source file to ingest plus the adapter that reads it."""

    path: str
    adapter: SourceAdapter


@dataclass
class IngestMetrics:
    """Observability counters (spec section 8), returned per pipeline run."""

    records_parsed: int = 0
    records_stored: int = 0
    duplicates_found: int = 0
    needs_review: int = 0
    needs_schema_review: int = 0
    requires_legal_review: int = 0
    errors: int = 0
    sources: list[str] = field(default_factory=list)

    def duplicate_rate(self) -> float:
        total = self.records_parsed
        return round(self.duplicates_found / total, 4) if total else 0.0

    def needs_review_rate(self) -> float:
        return round(self.needs_review / self.records_stored, 4) if self.records_stored else 0.0

    def as_dict(self) -> dict:
        return {
            "records_parsed": self.records_parsed,
            "records_stored": self.records_stored,
            "duplicates_found": self.duplicates_found,
            "duplicate_rate": self.duplicate_rate(),
            "needs_review": self.needs_review,
            "needs_review_rate": self.needs_review_rate(),
            "needs_schema_review": self.needs_schema_review,
            "requires_legal_review": self.requires_legal_review,
            "errors": self.errors,
            "sources": self.sources,
        }


class IngestionPipeline:
    def __init__(self, db_path: str | Path = "data/output/ingestion.db"):
        self.db_path = db_path

    def run(
        self,
        sources: list[IngestSource],
        output_json: str | Path | None = "data/output/unified_dataset.json",
        ingest_id: str | None = None,
    ) -> IngestMetrics:
        ingest_id = ingest_id or f"ingest-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        metrics = IngestMetrics()

        raw_records: list[CanonicalRecord] = []

        for source in sources:
            metrics.sources.append(source.path)
            try:
                for raw in source.adapter.parse(source.path):
                    try:
                        record = normalize(raw, ingest_id=ingest_id)
                        record = apply_schema_inference(record)
                        record = apply_license_policy(record)
                        raw_records.append(record)
                        metrics.records_parsed += 1
                    except Exception:  # noqa: BLE001 - per-record isolation
                        metrics.errors += 1
                        logger.exception("Failed to normalize record from %s", source.path)
            except Exception:  # noqa: BLE001 - adapter-level failure for this source
                metrics.errors += 1
                logger.exception("Failed to parse source %s", source.path)

        if not raw_records:
            logger.warning("No records parsed across %d source(s)", len(sources))
            return metrics

        # Fingerprinting needs the whole batch to fit the TF-IDF corpus.
        apply_fingerprints(raw_records)

        # De-duplication consolidates near/exact duplicates across the batch.
        surviving = consolidate(raw_records)
        metrics.duplicates_found = len(raw_records) - len(surviving)

        validate_batch(surviving)

        with Storage(self.db_path) as storage:
            storage.upsert_batch(raw_records)  # persist duplicates too, for audit
            metrics.records_stored = len(surviving)
            for record in surviving:
                if record.validation.needs_review:
                    metrics.needs_review += 1
                if record.validation.needs_schema_review:
                    metrics.needs_schema_review += 1
                if record.validation.requires_legal_review:
                    metrics.requires_legal_review += 1

            if output_json:
                export_unified_dataset(surviving, output_json)

        return metrics
