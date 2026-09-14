"""Tests for pattern_mining.storage."""
import numpy as np
import pytest

from pattern_mining.storage import (
    get_assigned_problem_ids,
    migrate_db,
    read_assignments,
    read_patterns,
    update_pattern_centroid,
    write_assignments,
    write_patterns,
)


def test_migration_and_crud(tmp_path):
    db_path = tmp_path / "test_ingestion.db"

    # Migration creates tables idempotently
    migrate_db(db_path)
    migrate_db(db_path)

    # Patterns
    test_centroid = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    patterns = [
        {
            "pattern_id": "pattern_0",
            "label": "binary-search / two-pointer",
            "size": 5,
            "representative": "prob-1",
            "top_tags": ["binary-search", "array"],
            "difficulty_json": {"easy": 3, "medium": 2},
            "centroid": test_centroid,
            "created_at": "2026-09-12T12:00:00Z",
        },
        {
            "pattern_id": "pattern_1",
            "label": "dynamic-programming / memoization",
            "size": 10,
            "representative": "prob-2",
            "top_tags": ["dp"],
            "difficulty_json": {"medium": 7, "hard": 3},
            "centroid": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            "created_at": "2026-09-12T12:00:00Z",
        },
    ]

    write_patterns(patterns, db_path)
    read_back = read_patterns(db_path)
    assert len(read_back) == 2
    p0 = next(p for p in read_back if p["pattern_id"] == "pattern_0")
    assert p0["label"] == "binary-search / two-pointer"
    assert p0["size"] == 5
    assert p0["representative"] == "prob-1"
    assert p0["top_tags"] == ["binary-search", "array"]
    assert p0["difficulty_distribution"] == {"easy": 3, "medium": 2}
    np.testing.assert_allclose(p0["centroid"], test_centroid, rtol=1e-5)

    # Assignments
    assignments = [
        ("prob-1", "pattern_0", 1.0),
        ("prob-2", "pattern_1", 0.95),
        ("prob-3", "pattern_0", 0.88, "2026-09-12T12:30:00Z"),
    ]
    write_assignments(assignments, db_path)
    read_as = read_assignments(db_path)
    assert len(read_as) == 3
    assigned_ids = get_assigned_problem_ids(db_path)
    assert assigned_ids == {"prob-1", "prob-2", "prob-3"}

    # Update centroid
    new_c = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
    update_pattern_centroid("pattern_0", new_c, 6, db_path)
    updated = read_patterns(db_path)
    p0_up = next(p for p in updated if p["pattern_id"] == "pattern_0")
    assert p0_up["size"] == 6
    np.testing.assert_allclose(p0_up["centroid"], new_c, rtol=1e-5)
