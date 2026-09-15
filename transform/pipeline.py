"""Template Pipeline orchestrator: runs `reauthor_template` over a batch
of canonical records, persisting accepted templates/variants and
returning observability metrics, mirroring `ingestion.pipeline`'s style
(`IngestMetrics`-style dataclass, `.as_dict()` for CLI JSON output).

Incremental by default: a canonical record is skipped if it already has
a template in `storage.templates` (looked up by `canonical_problem_id`),
matching the design doc's "only call transform on canonical problems
that do not already have an associated template_id" requirement. Pass
`force=True` to re-run and overwrite existing templates anyway.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .llm import LLMCallError, LLMClient, get_default_client
from .reauthor import DEFAULT_MAX_ATTEMPTS, reauthor_template
from .storage import (
    link_pattern_to_template,
    read_templates_for_canonical_problem,
    write_template,
    write_variants,
)
from .validator import DEFAULT_VARIANT_COUNT

logger = logging.getLogger(__name__)

_PROGRESS_EVERY = 10  # emit a running-totals line every N processed records


@dataclass
class TemplatePipelineMetrics:
    records_considered: int = 0
    records_skipped_existing: int = 0
    accepted: int = 0
    rejected_invalid_json: int = 0
    rejected_validation_failed: int = 0
    needs_review: int = 0
    llm_errors: int = 0
    total_variants_generated: int = 0
    rejections: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "records_considered": self.records_considered,
            "records_skipped_existing": self.records_skipped_existing,
            "accepted": self.accepted,
            "rejected_invalid_json": self.rejected_invalid_json,
            "rejected_validation_failed": self.rejected_validation_failed,
            "acceptance_rate": self.acceptance_rate(),
            "needs_review": self.needs_review,
            "llm_errors": self.llm_errors,
            "total_variants_generated": self.total_variants_generated,
        }

    def acceptance_rate(self) -> float:
        attempted = self.accepted + self.rejected_invalid_json + self.rejected_validation_failed
        return round(self.accepted / attempted, 4) if attempted else 0.0


def _load_unified_dataset(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pattern_assignments(db_path: str | Path) -> dict[str, str]:
    """Best-effort problem_id -> pattern_id lookup from pattern_mining's
    tables. Returns an empty dict (not an error) if pattern_mining has
    never run against this DB — templates just won't have a pattern
    label/pattern_id association until it has.
    """
    try:
        from pattern_mining.storage import read_assignments

        assignments = read_assignments(db_path)
        return {a["problem_id"]: a["pattern_id"] for a in assignments}
    except Exception:  # noqa: BLE001 - pattern_mining tables may not exist yet
        logger.info("No pattern_mining assignments found in %s; proceeding without pattern labels", db_path)
        return {}


def generate_templates_for_records(
    records: list[dict[str, Any]],
    db_path: str | Path = "data/output/ingestion.db",
    llm_client: Optional[LLMClient] = None,
    force: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    variant_count: int = DEFAULT_VARIANT_COUNT,
    llm_timeout: float = 300.0,
    llm_max_tokens: int = 512,
) -> TemplatePipelineMetrics:
    """Runs the reauthor pipeline over `records` (canonical record dicts,
    the shape found in unified_dataset.json), persisting every accepted
    template + its variants, and linking the template to its pattern
    (if a pattern_mining assignment exists for that canonical problem).

    Skips records with `data_unavailable=True` outright (no canonical
    solution exists to validate against, so `reauthor_template` would
    just reject them anyway) — filtered here rather than left to hit
    the validator, so metrics distinguish "skipped, known to be
    unvalidatable" from "attempted and failed validation".
    """
    llm_client = llm_client or get_default_client()
    pattern_by_problem_id = _load_pattern_assignments(db_path)
    metrics = TemplatePipelineMetrics()
    total = len(records)
    run_start = time.monotonic()

    logger.info(
        "transform pipeline starting: %d record(s) to process, LLM=%s, max_attempts=%d, variant_count=%d, timeout=%.0fs, max_tokens=%d",
        total, llm_client.__class__.__name__, max_attempts, variant_count, llm_timeout, llm_max_tokens,
    )

    for idx, record in enumerate(records, 1):
        canonical_id = record.get("id")
        if not canonical_id:
            continue

        metrics.records_considered += 1

        if not force and read_templates_for_canonical_problem(canonical_id, db_path):
            metrics.records_skipped_existing += 1
            logger.debug("[%d/%d] SKIP (already has template): %s", idx, total, canonical_id)
            continue

        if record.get("data_unavailable"):
            metrics.rejected_validation_failed += 1
            metrics.rejections.append(
                {"canonical_problem_id": canonical_id, "reason": "data_unavailable (no solution to validate against)"}
            )
            logger.info("[%d/%d] SKIP data_unavailable: %s", idx, total, canonical_id)
            continue

        pattern_label = pattern_by_problem_id.get(canonical_id)
        title = record.get("title", canonical_id)
        logger.info("[%d/%d] Processing: %s  (pattern=%s)", idx, total, title, pattern_label or "none")

        record_start = time.monotonic()
        try:
            result = reauthor_template(
                record,
                llm_client,
                pattern_label=pattern_label,
                max_attempts=max_attempts,
                variant_count=variant_count,
                timeout=llm_timeout,
                max_tokens=llm_max_tokens,
            )
        except LLMCallError as exc:
            metrics.llm_errors += 1
            metrics.rejections.append({"canonical_problem_id": canonical_id, "reason": f"LLM call failed: {exc}"})
            logger.error("[%d/%d] LLM ERROR (%.1fs): %s — %s", idx, total, time.monotonic() - record_start, canonical_id, exc)
            continue
        elapsed = time.monotonic() - record_start

        if result.status == "rejected_invalid_json":
            metrics.rejected_invalid_json += 1
            metrics.rejections.append({"canonical_problem_id": canonical_id, "reason": result.reasons})
            logger.warning(
                "[%d/%d] REJECTED invalid_json (%.1fs, %d attempt(s)): %s — %s",
                idx, total, elapsed, result.attempts, canonical_id, result.reasons,
            )
            continue

        if result.status == "rejected_validation_failed":
            metrics.rejected_validation_failed += 1
            metrics.rejections.append({"canonical_problem_id": canonical_id, "reason": result.reasons})
            logger.warning(
                "[%d/%d] REJECTED validation_failed (%.1fs): %s — %s",
                idx, total, elapsed, canonical_id, result.reasons,
            )
            continue

        # accepted
        assert result.template is not None
        if pattern_label and pattern_by_problem_id.get(canonical_id):
            result.template.pattern_id = pattern_by_problem_id[canonical_id]

        write_template(result.template, db_path)
        write_variants(result.variants, db_path)
        if result.template.pattern_id:
            link_pattern_to_template(result.template.pattern_id, result.template.template_id, db_path)

        metrics.accepted += 1
        metrics.total_variants_generated += len(result.variants)
        if result.template.needs_review:
            metrics.needs_review += 1
            logger.info(
                "[%d/%d] ACCEPTED needs_review (%.1fs, %d variant(s)): %s — review: %s",
                idx, total, elapsed, len(result.variants), canonical_id,
                result.template.review_reasons,
            )
        else:
            logger.info(
                "[%d/%d] ACCEPTED (%.1fs, %d variant(s)): %s",
                idx, total, elapsed, len(result.variants), canonical_id,
            )

        # Running totals every _PROGRESS_EVERY records
        if idx % _PROGRESS_EVERY == 0:
            attempted = metrics.accepted + metrics.rejected_invalid_json + metrics.rejected_validation_failed
            rate = f"{metrics.accepted / attempted:.0%}" if attempted else "n/a"
            elapsed_total = time.monotonic() - run_start
            avg_per_record = elapsed_total / idx
            eta_secs = avg_per_record * (total - idx)
            logger.info(
                "--- progress %d/%d | accepted=%d rejected_json=%d rejected_val=%d llm_err=%d "
                "| accept_rate=%s | elapsed=%.0fs ETA≈%.0fs ---",
                idx, total,
                metrics.accepted, metrics.rejected_invalid_json,
                metrics.rejected_validation_failed, metrics.llm_errors,
                rate, elapsed_total, eta_secs,
            )

    total_elapsed = time.monotonic() - run_start
    attempted = metrics.accepted + metrics.rejected_invalid_json + metrics.rejected_validation_failed
    logger.info(
        "transform pipeline finished: %d considered, %d accepted, %d rejected_json, "
        "%d rejected_val, %d llm_errors, %d skipped | accept_rate=%s | total=%.1fs",
        metrics.records_considered, metrics.accepted, metrics.rejected_invalid_json,
        metrics.rejected_validation_failed, metrics.llm_errors, metrics.records_skipped_existing,
        f"{metrics.accepted / attempted:.0%}" if attempted else "n/a",
        total_elapsed,
    )

    return metrics


def generate_templates_for_patterns(
    unified_json_path: str | Path = "data/output/unified_dataset.json",
    db_path: str | Path = "data/output/ingestion.db",
    llm_client: Optional[LLMClient] = None,
    limit: Optional[int] = None,
    force: bool = False,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    variant_count: int = DEFAULT_VARIANT_COUNT,
    llm_timeout: float = 300.0,
    llm_max_tokens: int = 512,
) -> TemplatePipelineMetrics:
    """Entry point matching the design doc's
    `transform.template_pipeline.generate_templates_for_patterns` call
    site: loads the unified dataset from disk and runs the template
    pipeline over it (optionally capped to `limit` records — useful for
    smoke-testing against a small slice before a full batch run, since
    each record costs one LLM call plus 1-8 subprocess executions).
    """
    records = _load_unified_dataset(unified_json_path)
    if limit is not None:
        records = records[:limit]

    return generate_templates_for_records(
        records,
        db_path=db_path,
        llm_client=llm_client,
        force=force,
        max_attempts=max_attempts,
        variant_count=variant_count,
        llm_timeout=llm_timeout,
        llm_max_tokens=llm_max_tokens,
    )
