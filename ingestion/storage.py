"""Storage & Versioning.

Persists canonical records to SQLite (standing in for Postgres+JSONB per
spec section 3.7 — same idea: a table keyed by id with a JSON payload
column, queryable idempotently). Also exports the unified dataset as
`unified_dataset.json` for downstream consumers that just want a flat
file.

Idempotency (spec section 9): each record's `ingest_trace` carries
`raw_file` + `raw_offset`. We upsert keyed on
`(source_name, raw_file, raw_offset)` so re-running the same source file
updates the existing record in place (bumping version + history) instead
of creating a duplicate row.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import CanonicalRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    raw_file TEXT,
    raw_offset INTEGER,
    title TEXT NOT NULL,
    deduplicated_into TEXT,
    version INTEGER NOT NULL,
    needs_review INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_source_trace
    ON records (source_name, raw_file, raw_offset);

CREATE TABLE IF NOT EXISTS ingest_cursor (
    source_key TEXT PRIMARY KEY,
    last_offset INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _find_existing_id(self, record: CanonicalRecord) -> str | None:
        trace = record.ingest_trace
        if trace is None or not record.source_list:
            return None
        cur = self.conn.execute(
            "SELECT id FROM records WHERE source_name = ? AND raw_file = ? AND raw_offset = ?",
            (record.source_list[0], trace.raw_file, trace.raw_offset),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def upsert(self, record: CanonicalRecord) -> CanonicalRecord:
        """Idempotent upsert keyed by (source_name, raw_file, raw_offset).
        If a matching record already exists, reuses its id and bumps
        version/history rather than inserting a new row.
        """
        existing_id = self._find_existing_id(record)
        if existing_id is not None:
            record.id = existing_id
            record.bump_version("re-ingested", detail="matched existing raw_file/raw_offset")

        trace = record.ingest_trace
        self.conn.execute(
            """
            INSERT INTO records (id, source_name, raw_file, raw_offset, title, deduplicated_into,
                                  version, needs_review, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                deduplicated_into=excluded.deduplicated_into,
                version=excluded.version,
                needs_review=excluded.needs_review,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                record.id,
                record.source_list[0] if record.source_list else "unknown",
                trace.raw_file if trace else None,
                trace.raw_offset if trace else None,
                record.title,
                record.deduplicated_into,
                record.version,
                int(record.validation.needs_review),
                record.model_dump_json(),
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
            ),
        )
        return record

    def upsert_batch(self, records: list[CanonicalRecord]) -> list[CanonicalRecord]:
        for record in records:
            self.upsert(record)
        self.conn.commit()
        return records

    def all_records(self, include_deduplicated: bool = False) -> list[CanonicalRecord]:
        query = "SELECT payload FROM records"
        if not include_deduplicated:
            query += " WHERE deduplicated_into IS NULL"
        cur = self.conn.execute(query)
        return [CanonicalRecord.model_validate_json(row[0]) for row in cur.fetchall()]

    def get_cursor(self, source_key: str) -> int:
        cur = self.conn.execute(
            "SELECT last_offset FROM ingest_cursor WHERE source_key = ?", (source_key,)
        )
        row = cur.fetchone()
        return row[0] if row else -1

    def set_cursor(self, source_key: str, offset: int) -> None:
        from datetime import datetime, timezone

        self.conn.execute(
            """
            INSERT INTO ingest_cursor (source_key, last_offset, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                last_offset=excluded.last_offset, updated_at=excluded.updated_at
            """,
            (source_key, offset, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()


def export_unified_dataset(records: list[CanonicalRecord], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(r.model_dump_json()) for r in records]
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
