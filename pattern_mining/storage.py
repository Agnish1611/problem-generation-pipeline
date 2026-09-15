"""Database read/write utilities and schema migrations for pattern mining.

Manages the persistence of algorithmic patterns and problem assignments
into the SQLite ingestion database, using atomic transactions and safe IO.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

SCHEMA_MIGRATION = """
CREATE TABLE IF NOT EXISTS patterns (
  pattern_id TEXT PRIMARY KEY,
  label TEXT,
  size INTEGER,
  representative TEXT,
  top_tags TEXT,         -- JSON string
  difficulty_json TEXT,  -- JSON string
  centroid BLOB,         -- BLOB containing numpy float32 bytes
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS pattern_assignments (
  problem_id TEXT PRIMARY KEY,
  pattern_id TEXT,
  confidence REAL,
  assigned_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_pattern_assignments_pattern_id
  ON pattern_assignments (pattern_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def migrate_db(db_path: str | Path) -> None:
    """Run idempotent schema migrations for pattern mining tables."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_MIGRATION)
        conn.commit()


def _serialize_centroid(centroid: Any) -> bytes:
    """Convert array or list to float32 binary blob."""
    if isinstance(centroid, bytes):
        return centroid
    arr = np.asarray(centroid, dtype=np.float32)
    return arr.tobytes()


def _deserialize_centroid(blob: bytes | None) -> np.ndarray | None:
    """Convert binary blob back to 1D float32 numpy array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


def write_patterns(patterns: Sequence[dict[str, Any]], db_path: str | Path) -> None:
    """Upsert a list of pattern dictionaries into the patterns table.

    Each pattern dict may contain:
      pattern_id, label, size (or count), representative, top_tags,
      difficulty_json (or difficulty_distribution), centroid (or centroid_embedding),
      created_at
    """
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        records = []
        for p in patterns:
            pattern_id = p["pattern_id"]
            label = p.get("label", "")
            size = p.get("size", p.get("count", 0))
            rep = p.get("representative", "")

            top_tags = p.get("top_tags", [])
            if not isinstance(top_tags, str):
                top_tags = json.dumps(top_tags)

            diff_dist = p.get("difficulty_json", p.get("difficulty_distribution", {}))
            if not isinstance(diff_dist, str):
                diff_dist = json.dumps(diff_dist)

            raw_centroid = p.get("centroid", p.get("centroid_embedding"))
            centroid_blob = _serialize_centroid(raw_centroid) if raw_centroid is not None else None
            created_at = p.get("created_at", _now_iso())

            records.append((
                pattern_id, label, size, rep, top_tags, diff_dist, centroid_blob, created_at
            ))

        conn.executemany(
            """
            INSERT INTO patterns (pattern_id, label, size, representative, top_tags, difficulty_json, centroid, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern_id) DO UPDATE SET
                label = excluded.label,
                size = excluded.size,
                representative = excluded.representative,
                top_tags = excluded.top_tags,
                difficulty_json = excluded.difficulty_json,
                centroid = excluded.centroid,
                created_at = excluded.created_at
            """,
            records,
        )
        conn.commit()


def write_assignments(
    assignments: Sequence[Tuple[Any, ...]],
    db_path: str | Path,
) -> None:
    """Upsert problem-pattern assignments into pattern_assignments table.

    Assignments can be tuples of (problem_id, pattern_id, confidence)
    or (problem_id, pattern_id, confidence, assigned_at).
    """
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        records = []
        for item in assignments:
            if len(item) == 3:
                problem_id, pattern_id, confidence = item  # type: ignore[misc]
                assigned_at = _now_iso()
            else:
                problem_id, pattern_id, confidence, assigned_at = item  # type: ignore[misc]
            records.append((str(problem_id), str(pattern_id), float(confidence), str(assigned_at)))

        conn.executemany(
            """
            INSERT INTO pattern_assignments (problem_id, pattern_id, confidence, assigned_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(problem_id) DO UPDATE SET
                pattern_id = excluded.pattern_id,
                confidence = excluded.confidence,
                assigned_at = excluded.assigned_at
            """,
            records,
        )
        conn.commit()


def read_patterns(db_path: str | Path) -> List[dict[str, Any]]:
    """Read all patterns from patterns table, deserializing JSON and centroid blobs."""
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT pattern_id, label, size, representative, top_tags, difficulty_json, centroid, created_at FROM patterns ORDER BY pattern_id"
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            top_tags = json.loads(r["top_tags"]) if r["top_tags"] else []
            diff_dist = json.loads(r["difficulty_json"]) if r["difficulty_json"] else {}
            centroid = _deserialize_centroid(r["centroid"])
            results.append({
                "pattern_id": r["pattern_id"],
                "label": r["label"],
                "size": r["size"],
                "representative": r["representative"],
                "top_tags": top_tags,
                "difficulty_distribution": diff_dist,
                "difficulty_json": r["difficulty_json"],
                "centroid": centroid,
                "centroid_embedding": centroid.tolist() if centroid is not None else [],
                "created_at": r["created_at"],
            })
        return results


def read_assignments(db_path: str | Path) -> List[dict[str, Any]]:
    """Read all assignments from pattern_assignments table."""
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT problem_id, pattern_id, confidence, assigned_at FROM pattern_assignments ORDER BY problem_id"
        )
        return [
            {
                "problem_id": r["problem_id"],
                "pattern_id": r["pattern_id"],
                "confidence": r["confidence"],
                "assigned_at": r["assigned_at"],
            }
            for r in cursor.fetchall()
        ]


def get_assigned_problem_ids(db_path: str | Path) -> Set[str]:
    """Return the set of problem IDs that already have pattern assignments."""
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT problem_id FROM pattern_assignments")
        return {r["problem_id"] for r in cursor.fetchall()}


def get_assigned_content_hashes(db_path: str | Path) -> Set[str]:
    """Return set of content_hashes for problems that already have assignments.

    Cross-references pattern_assignments with the records table if available.
    """
    migrate_db(db_path)
    with get_connection(db_path) as conn:
        # Check if records table exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='records'"
        )
        if not cur.fetchone():
            return set()

        cursor = conn.execute(
            """
            SELECT json_extract(r.payload, '$.fingerprints.content_hash') as ch
            FROM records r
            JOIN pattern_assignments pa ON r.id = pa.problem_id
            WHERE ch IS NOT NULL
            """
        )
        return {r["ch"] for r in cursor.fetchall() if r["ch"]}


def update_pattern_centroid(
    pattern_id: str,
    new_centroid: np.ndarray,
    new_size: int,
    db_path: str | Path,
) -> None:
    """Update centroid and size of an existing pattern (atomic update)."""
    migrate_db(db_path)
    centroid_blob = _serialize_centroid(new_centroid)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE patterns
            SET centroid = ?, size = ?
            WHERE pattern_id = ?
            """,
            (centroid_blob, new_size, pattern_id),
        )
        conn.commit()
