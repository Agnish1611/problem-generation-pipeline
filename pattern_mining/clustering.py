"""Clustering orchestration using UMAP and HDBSCAN.

Reduces embedding dimensions with UMAP (seeded for determinism) and discovers
algorithmic patterns using HDBSCAN with cosine metric.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: np.ndarray,
    ids: List[str],
    umap_dim: int = 64,
    min_cluster_size: int = 10,
    min_samples: int = 1,
    random_state: int = 42,
    metric: str = "cosine",
    cluster_selection_method: str = "eom",
    assign_noise_threshold: Optional[float] = 0.75,
) -> np.ndarray:
    """Cluster embeddings into algorithmic patterns.

    - Uses UMAP to reduce dimensions to `umap_dim` only if embeddings.shape[1] > umap_dim.
    - Uses HDBSCAN with metric='cosine' and cluster_selection_method='eom'.
    - Optionally reassigns noise points (-1) with cosine similarity >= assign_noise_threshold
      to their nearest cluster centroid.
    - Returns labels (numpy ndarray of int), where -1 indicates noise/outliers.
    - Logs cluster counts and outlier count.
    """
    n_samples, n_features = embeddings.shape
    if n_samples == 0:
        logger.warning("Empty embeddings array passed to cluster_embeddings.")
        return np.empty(0, dtype=int)

    if n_samples < min_cluster_size:
        logger.warning(
            "n_samples (%d) is smaller than min_cluster_size (%d); all labeled as noise (-1).",
            n_samples,
            min_cluster_size,
        )
        return np.full(n_samples, -1, dtype=int)

    # 1. Dimension reduction via UMAP if n_features > umap_dim
    effective_dim = min(umap_dim, max(2, n_samples - 2))
    if n_features > effective_dim and n_samples >= 4:
        logger.info(
            "Reducing embedding dimensions from %d to %d using UMAP (metric=%s, random_state=%d)...",
            n_features,
            effective_dim,
            metric,
            random_state,
        )
        import umap

        # Handle small n_samples gracefully
        n_neighbors = min(15, max(2, n_samples - 1))
        init_method = "random" if n_samples < 20 else "spectral"
        reducer = umap.UMAP(
            n_components=effective_dim,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            init=init_method,
            random_state=random_state,
            transform_seed=random_state,
        )
        reduced_embeddings: np.ndarray = np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
        logger.info("UMAP reduction complete. Reduced shape: %s", reduced_embeddings.shape)
    else:
        logger.info("Skipping UMAP: embedding features (%d) <= effective_dim (%d)", n_features, effective_dim)
        reduced_embeddings = embeddings

    # 2. HDBSCAN clustering
    logger.info(
        "Clustering with HDBSCAN (metric=%s, cluster_selection_method=%s, min_cluster_size=%d, min_samples=%d)...",
        metric,
        cluster_selection_method,
        min_cluster_size,
        min_samples,
    )

    labels = _run_hdbscan(
        reduced_embeddings=reduced_embeddings,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    # 3. Soft-assign borderline noise points if requested
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    if assign_noise_threshold is not None and unique_clusters and np.any(labels == -1):
        noise_mask = (labels == -1)
        # Compute centroids in original embedding space
        centroids = np.array([np.mean(embeddings[labels == c], axis=0) for c in unique_clusters])
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1.0
        norm_centroids = centroids / c_norms

        noise_embs = embeddings[noise_mask]
        n_norms = np.linalg.norm(noise_embs, axis=1, keepdims=True)
        n_norms[n_norms == 0] = 1.0
        norm_noise = noise_embs / n_norms

        sims = np.dot(norm_noise, norm_centroids.T)
        best_cluster_indices = np.argmax(sims, axis=1)
        best_sims = np.max(sims, axis=1)

        reassigned_count = 0
        noise_indices = np.where(noise_mask)[0]
        for idx_in_noise, best_sim in enumerate(best_sims):
            if best_sim >= assign_noise_threshold:
                target_cluster = unique_clusters[best_cluster_indices[idx_in_noise]]
                labels[noise_indices[idx_in_noise]] = target_cluster
                reassigned_count += 1

        if reassigned_count > 0:
            logger.info(
                "Reassigned %d borderline noise points to nearest cluster (similarity >= %.2f)",
                reassigned_count,
                assign_noise_threshold,
            )

    unique_labels = set(labels)
    n_clusters = len([l for l in unique_labels if l != -1])
    n_outliers = int(np.sum(labels == -1))
    outlier_pct = (n_outliers / n_samples) * 100 if n_samples > 0 else 0.0

    logger.info(
        "Clustering finished: %d records, %d clusters discovered, %d outliers (%.1f%%)",
        n_samples,
        n_clusters,
        n_outliers,
        outlier_pct,
    )

    return labels


def _run_hdbscan(
    reduced_embeddings: np.ndarray,
    metric: str,
    cluster_selection_method: str,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    """Run HDBSCAN with cosine metric, supporting sklearn.cluster.HDBSCAN and hdbscan fallback."""
    # Attempt 1: sklearn.cluster.HDBSCAN (supports metric='cosine' natively)
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        clusterer = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        return clusterer.fit_predict(reduced_embeddings)
    except Exception as err:
        logger.debug("sklearn.cluster.HDBSCAN failed (%s), falling back to hdbscan package", err)

    # Attempt 2: hdbscan package
    import hdbscan

    if metric == "cosine":
        # hdbscan package expects precomputed distances for cosine
        dist_matrix = cosine_distances(reduced_embeddings).astype(np.float64)
        # Numerical guard: clip distance matrix to [0, 2]
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            cluster_selection_method=cluster_selection_method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        return clusterer.fit_predict(dist_matrix)
    else:
        clusterer = hdbscan.HDBSCAN(
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        return clusterer.fit_predict(reduced_embeddings.astype(np.float64))


def compute_clusters_from_unified(
    unified_json_path: str | Path = "data/output/unified_dataset.json",
    cache_dir: str | Path = "data/processed",
    recompute_embeddings: bool = False,
    **cluster_params: Any,
) -> Dict[str, int]:
    """Load unified dataset, compute/load embeddings, run clustering, and return problem_id -> label mapping."""
    from .embeddings import compute_all_embeddings

    embeddings, ids = compute_all_embeddings(
        unified_json_path=unified_json_path,
        cache_dir=cache_dir,
        recompute=recompute_embeddings,
    )

    labels = cluster_embeddings(embeddings, ids, **cluster_params)
    return {pid: int(lbl) for pid, lbl in zip(ids, labels)}
