"""Persistence for the re-authoring pipeline: `templates` and `variants`
tables in the shared ingestion.db SQLite file, plus a `template_id`
column added to pattern_mining's `patterns` table so a pattern can point
at its representative template.

Follows the same idempotent-migration style as `pattern_mining/storage.py`
(`migrate_db` run at the top of every public function, `INSERT ... ON
CONFLICT DO UPDATE` upserts) rather than `ingestion/storage.py`'s
class-based `Storage` — templates/variants don't need the raw_file/offset
idempotency key ingestion uses, since they're keyed by `template_id`/
`variant_id` directly.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .models import Template, Variant

SCHEMA_MIGRATION = """
CREATE TABLE IF NOT EXISTS templates (
  template_id TEXT PRIMARY KEY,
  canonical_problem_id TEXT,
  pattern_id TEXT,
  title TEXT,
  description TEXT,
  input_schema TEXT,     -- JSON string
  output_schema TEXT,    -- JSON string
  constraints TEXT,      -- JSON string
  tie_breaker TEXT,
  transform_seed TEXT,
  transform_model TEXT,
  signature_hash TEXT,
  needs_review INTEGER NOT NULL DEFAULT 0,
  review_reasons TEXT,   -- JSON string
  provenance TEXT,       -- JSON string: {source_list, doocs_commit_hash, license, ...}
  created_at TEXT,
  updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_templates_canonical_problem_id
  ON templates (canonical_problem_id);

CREATE TABLE IF NOT EXISTS variants (
  variant_id TEXT PRIMARY KEY,
  template_id TEXT,
  seed TEXT,
  args TEXT,              -- JSON string
  canonical_output TEXT,  -- JSON string
  signature_hash TEXT,
  created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_variants_template_id
  ON variants (template_id);
"""

# patterns.template_id: added via a separate idempotent ALTER TABLE
# rather than baked into pattern_mining's own SCHEMA_MIGRATION, since
# that table is owned by pattern_mining/storage.py and this module
# shouldn't need to duplicate its full CREATE TABLE statement just to
# add one nullable column.
_ADD_PATTERN_TEMPLATE_ID_COLUMN = "ALTER TABLE patterns ADD COLUMN template_id TEXT"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def migrate_db(db_path: str | Path) -> None:
    """Idempotent schema migration: creates templates/variants tables if
    missing, and adds patterns.template_id if the patterns table exists
    and doesn't already have that column. Safe to call before
    pattern_mining has ever run (patterns table absent) — the ALTER TABLE
    is skipped in that case, not treated as an error.
    """
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_MIGRATION)

        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patterns'")
        if cur.fetchone():
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(patterns)")}
            if "template_id" not in existing_cols:
                conn.execute(_ADD_PATTERN_TEMPLATE_ID_COLUMN)

        conn.commit()


def write_template(template: Template, db_path: str | Path) -> None:
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO templates (
                template_id, canonical_problem_id, pattern_id, title, description,
                input_schema, output_schema, constraints, tie_breaker, transform_seed,
                transform_model, signature_hash, needs_review, review_reasons, provenance,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(template_id) DO UPDATE SET
                canonical_problem_id = excluded.canonical_problem_id,
                pattern_id = excluded.pattern_id,
                title = excluded.title,
                description = excluded.description,
                input_schema = excluded.input_schema,
                output_schema = excluded.output_schema,
                constraints = excluded.constraints,
                tie_breaker = excluded.tie_breaker,
                transform_seed = excluded.transform_seed,
                transform_model = excluded.transform_model,
                signature_hash = excluded.signature_hash,
                needs_review = excluded.needs_review,
                review_reasons = excluded.review_reasons,
                provenance = excluded.provenance,
                updated_at = excluded.updated_at
            """,
            (
                template.template_id,
                template.canonical_problem_id,
                template.pattern_id,
                template.title,
                template.description,
                json.dumps(template.input_schema),
                json.dumps(template.output_schema),
                json.dumps(template.constraints),
                template.tie_breaker,
                template.transform_seed,
                template.transform_model,
                template.signature_hash,
                int(template.needs_review),
                json.dumps(template.review_reasons),
                template.provenance.model_dump_json(),
                template.created_at.isoformat(),
                _now_iso(),
            ),
        )
        conn.commit()


def write_variants(variants: Sequence[Variant], db_path: str | Path) -> None:
    if not variants:
        return
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO variants (variant_id, template_id, seed, args, canonical_output, signature_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(variant_id) DO UPDATE SET
                template_id = excluded.template_id,
                seed = excluded.seed,
                args = excluded.args,
                canonical_output = excluded.canonical_output,
                signature_hash = excluded.signature_hash
            """,
            [
                (
                    v.variant_id,
                    v.template_id,
                    v.seed,
                    json.dumps(v.args),
                    json.dumps(v.canonical_output),
                    v.signature_hash,
                    v.created_at.isoformat(),
                )
                for v in variants
            ],
        )
        conn.commit()


def link_pattern_to_template(pattern_id: str, template_id: str, db_path: str | Path) -> None:
    """Sets patterns.template_id for an existing pattern row. No-op
    (does not raise) if the patterns table or the given pattern_id
    doesn't exist yet — pattern_mining may not have run, or the pattern
    may have been dropped/renumbered by a later re-clustering run.
    """
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patterns'")
        if not cur.fetchone():
            return
        conn.execute("UPDATE patterns SET template_id = ? WHERE pattern_id = ?", (template_id, pattern_id))
        conn.commit()


def read_template(template_id: str, db_path: str | Path) -> dict[str, Any] | None:
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM templates WHERE template_id = ?", (template_id,)).fetchone()
        if row is None:
            return None
        return _row_to_template_dict(row)


def read_templates_for_canonical_problem(canonical_problem_id: str, db_path: str | Path) -> list[dict[str, Any]]:
    """Used by the pipeline entry point to skip canonical problems that
    already have a template (incremental-mode gate)."""
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM templates WHERE canonical_problem_id = ?", (canonical_problem_id,)
        ).fetchall()
        return [_row_to_template_dict(r) for r in rows]


def read_variants_for_template(template_id: str, db_path: str | Path) -> list[dict[str, Any]]:
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM variants WHERE template_id = ?", (template_id,)).fetchall()
        return [
            {
                "variant_id": r["variant_id"],
                "template_id": r["template_id"],
                "seed": r["seed"],
                "args": json.loads(r["args"]) if r["args"] else {},
                "canonical_output": json.loads(r["canonical_output"]) if r["canonical_output"] else None,
                "signature_hash": r["signature_hash"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


def _row_to_template_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "template_id": row["template_id"],
        "canonical_problem_id": row["canonical_problem_id"],
        "pattern_id": row["pattern_id"],
        "title": row["title"],
        "description": row["description"],
        "input_schema": json.loads(row["input_schema"]) if row["input_schema"] else {},
        "output_schema": json.loads(row["output_schema"]) if row["output_schema"] else {},
        "constraints": json.loads(row["constraints"]) if row["constraints"] else [],
        "tie_breaker": row["tie_breaker"],
        "transform_seed": row["transform_seed"],
        "transform_model": row["transform_model"],
        "signature_hash": row["signature_hash"],
        "needs_review": bool(row["needs_review"]),
        "review_reasons": json.loads(row["review_reasons"]) if row["review_reasons"] else [],
        "provenance": json.loads(row["provenance"]) if row["provenance"] else {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
