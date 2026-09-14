"""Tests for pattern_mining.assigner."""
import numpy as np
import pytest

from pattern_mining.assigner import (
    CentroidTracker,
    assign_new_problem,
    bulk_assign_new_records,
)
from pattern_mining.storage import read_assignments, read_patterns, write_patterns


class MockModelForAssigner:
    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        res = []
        for t in texts:
            # find matching vector or return default
            vec = None
            for key, v in self.mapping.items():
                if key in t:
                    vec = v
                    break
            if vec is None:
                vec = np.zeros(16, dtype=np.float32)
                vec[0] = 1.0
            res.append(vec)
        arr = np.array(res, dtype=np.float32)
        if len(texts) == 1 and not kwargs.get("batch_size"):
            return arr[0]
        return arr


def test_assigner_assigns_to_nearest_pattern(tmp_path):
    db_path = tmp_path / "test.db"

    # Pattern 0 centroid: points along axis 0
    c0 = np.zeros(16, dtype=np.float32)
    c0[0] = 1.0

    # Pattern 1 centroid: points along axis 1
    c1 = np.zeros(16, dtype=np.float32)
    c1[1] = 1.0

    initial_patterns = [
        {
            "pattern_id": "pattern_0",
            "label": "Pattern Zero",
            "size": 2,
            "representative": "p0",
            "top_tags": ["tag0"],
            "difficulty_json": {"easy": 2},
            "centroid": c0,
            "created_at": "2026-09-12T00:00:00Z",
        },
        {
            "pattern_id": "pattern_1",
            "label": "Pattern One",
            "size": 5,
            "representative": "p1",
            "top_tags": ["tag1"],
            "difficulty_json": {"medium": 5},
            "centroid": c1,
            "created_at": "2026-09-12T00:00:00Z",
        },
    ]
    write_patterns(initial_patterns, db_path)

    # Problem very similar to pattern 0
    prob0_vec = np.zeros(16, dtype=np.float32)
    prob0_vec[0] = 0.99
    prob0_vec[2] = 0.05
    prob0_vec = prob0_vec / np.linalg.norm(prob0_vec)

    # Problem very dissimilar to both (points along axis 5)
    prob_new_vec = np.zeros(16, dtype=np.float32)
    prob_new_vec[5] = 1.0

    mock_model = MockModelForAssigner({
        "Similar to Zero": prob0_vec,
        "Completely New Problem": prob_new_vec,
    })

    # Test 1: Assign problem similar to pattern 0
    new_prob = {
        "id": "p_sim",
        "title": "Similar to Zero",
        "description": "Some description",
    }
    pat_id, conf = assign_new_problem(
        new_prob,
        db_path=db_path,
        model=mock_model,
        threshold=0.80,
    )
    assert pat_id == "pattern_0"
    assert conf > 0.95

    # Verify centroid running average was updated:
    # old_size = 2, old_c0 = [1.0, 0, ...], new_emb = prob0_vec
    # expected_c = (2 * old_c0 + prob0_vec) / 3
    patterns_after = read_patterns(db_path)
    p0_after = next(p for p in patterns_after if p["pattern_id"] == "pattern_0")
    assert p0_after["size"] == 3
    expected_c0 = (c0 * 2.0 + prob0_vec) / 3.0
    np.testing.assert_allclose(p0_after["centroid"], expected_c0, rtol=1e-4)

    # Test 2: Problem with similarity < 0.80 creates new pattern_2
    new_prob_2 = {
        "id": "p_diff",
        "title": "Completely New Problem",
        "description": "Another description",
    }
    pat_id_2, conf_2 = assign_new_problem(
        new_prob_2,
        db_path=db_path,
        model=mock_model,
        threshold=0.80,
    )
    assert pat_id_2 == "pattern_2"
    assert conf_2 == 1.0

    patterns_final = read_patterns(db_path)
    assert len(patterns_final) == 3

    # Test 3: Bulk assign with duplicates skipped
    bulk_results = bulk_assign_new_records(
        [new_prob, new_prob_2],  # both already assigned!
        db_path=db_path,
        model=mock_model,
    )
    assert len(bulk_results) == 0  # skipped
