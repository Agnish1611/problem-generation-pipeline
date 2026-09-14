"""Tests for pattern_mining.clustering."""
import numpy as np
import pytest

from pattern_mining.clustering import cluster_embeddings


def test_cluster_embeddings_synthetic():
    # Synthetic dataset: 6 vectors in 16 dimensions
    # Two well-separated, tight clusters:
    # Cluster A: items 0, 1, 2
    # Cluster B: items 3, 4, 5
    rng = np.random.RandomState(42)

    c1_base = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    c2_base = np.array([0.0, 1.0, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)

    c1 = c1_base + rng.normal(0, 0.01, size=(3, 16)).astype(np.float32)
    c2 = c2_base + rng.normal(0, 0.01, size=(3, 16)).astype(np.float32)

    # Normalize vectors
    c1 = c1 / np.linalg.norm(c1, axis=1, keepdims=True)
    c2 = c2 / np.linalg.norm(c2, axis=1, keepdims=True)

    embeddings = np.vstack([c1, c2])
    ids = [f"prob_{i}" for i in range(6)]

    # umap_dim=8 (n_features=16 > umap_dim=8, so UMAP runs)
    labels = cluster_embeddings(
        embeddings=embeddings,
        ids=ids,
        umap_dim=8,
        min_cluster_size=2,
        min_samples=1,
        random_state=42,
    )

    assert len(labels) == 6
    # Check that items 0, 1, 2 have the same non-noise label
    assert labels[0] != -1
    assert labels[0] == labels[1] == labels[2]

    # Check that items 3, 4, 5 have the same non-noise label
    assert labels[3] != -1
    assert labels[3] == labels[4] == labels[5]

    # Check that the two clusters are distinct
    assert labels[0] != labels[3]
    unique_clusters = set(labels)
    assert len(unique_clusters) == 2
