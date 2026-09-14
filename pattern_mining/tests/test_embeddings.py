"""Tests for pattern_mining.embeddings."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pattern_mining.embeddings import (
    compose_text,
    compute_all_embeddings,
    get_text_embedding,
    load_embeddings,
)


class MockSentenceTransformer:
    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        # Return deterministic dummy embeddings based on length
        arr = []
        for t in texts:
            val = float(len(t) % 10) / 10.0
            vec = np.full(self.dim, val, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            arr.append(vec)
        res = np.array(arr, dtype=np.float32)
        if kwargs.get("convert_to_numpy", True):
            if len(texts) == 1 and not kwargs.get("batch_size"):
                return res[0]
            return res
        return res


def test_compose_text():
    prob = {
        "title": "Two Sum",
        "description": "Find indices of two numbers.",
        "examples": [
            {"input": "[2, 7], 9", "output": "[0, 1]"},
            {"input": "[3, 2, 4], 6", "output": "[1, 2]"},
            {"input": "[3, 3], 6", "output": "[0, 1]"},  # 3rd example should be ignored
        ],
    }
    composed = compose_text(prob)
    assert "Two Sum" in composed
    assert "Find indices of two numbers." in composed
    assert "Input: [2, 7], 9\nOutput: [0, 1]" in composed
    assert "Input: [3, 2, 4], 6\nOutput: [1, 2]" in composed
    assert "[3, 3]" not in composed


def test_get_text_embedding():
    mock_model = MockSentenceTransformer(dim=32)
    prob = {"id": "p1", "title": "Binary Search", "description": "Search target in sorted array."}
    emb = get_text_embedding(prob, model=mock_model)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (32,)
    assert emb.dtype == np.float32
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)


def test_compute_all_embeddings_and_cache(tmp_path):
    cache_dir = tmp_path / "processed"
    mock_model = MockSentenceTransformer(dim=16)

    problems = [
        {"id": "p1", "title": "Problem 1", "description": "Desc 1"},
        {"id": "p2", "title": "Problem 2", "description": "Desc 2"},
        {"id": "", "title": "Missing ID", "description": "Desc"},  # Should be skipped
        {"id": "p3", "description": "Missing title"},  # Should be skipped
    ]

    embs, ids = compute_all_embeddings(
        problems=problems,
        cache_dir=cache_dir,
        recompute=True,
        model=mock_model,
    )

    assert len(ids) == 2
    assert ids == ["p1", "p2"]
    assert embs.shape == (2, 16)

    # Check files created
    assert (cache_dir / "embeddings.npy").exists()
    assert (cache_dir / "ids.json").exists()
    assert (cache_dir / "skipped.json").exists()

    skipped = json.loads((cache_dir / "skipped.json").read_text())
    assert len(skipped) == 2

    # Load from cache without model
    loaded_embs, loaded_ids = load_embeddings(cache_dir=cache_dir)
    assert loaded_ids == ids
    np.testing.assert_allclose(loaded_embs, embs)

    # Recompute=False should return cached directly
    embs2, ids2 = compute_all_embeddings(
        problems=[],  # even with empty input, cache is used
        cache_dir=cache_dir,
        recompute=False,
    )
    assert ids2 == ids
    np.testing.assert_allclose(embs2, embs)
