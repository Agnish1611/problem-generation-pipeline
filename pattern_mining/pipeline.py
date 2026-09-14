"""Pipeline orchestrator for algorithmic pattern mining.

Coordinates: Loading Unified Dataset -> Embedding Computation/Caching ->
UMAP+HDBSCAN Clustering -> Cluster Summarization -> DB Persistence ->
patterns.json Export.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .clustering import cluster_embeddings
from .embeddings import compute_all_embeddings
from .storage import migrate_db, write_assignments, write_patterns
from .summarizer import summarize_clusters

logger = logging.getLogger(__name__)


def _atomic_write_json(filepath: Path, data: Any) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_name(f".tmp_{os.getpid()}_{filepath.name}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def run_full_pipeline(
    unified_json_path: str | Path = "data/output/unified_dataset.json",
    db_path: str | Path = "data/output/ingestion.db",
    output_patterns_path: str | Path = "data/output/patterns.json",
    cache_dir: str | Path = "data/processed",
    recompute_embeddings: bool = False,
    cluster_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the complete end-to-end pattern mining pipeline.

    1. Load canonical problems from unified dataset.
    2. Compute or load cached sentence embeddings.
    3. Run UMAP + HDBSCAN clustering.
    4. Summarize clusters into structured algorithmic patterns.
    5. Persist patterns and cluster assignments (confidence 1.0) to SQLite DB.
    6. Export data/output/patterns.json.
    7. Print and return operational metrics: n_records, n_clusters, outliers.
    """
    logger.info("=== Starting Pattern Mining Pipeline ===")
    start_time = datetime.now(timezone.utc)

    unified_path = Path(unified_json_path)
    if not unified_path.exists():
        raise FileNotFoundError(f"Unified dataset not found at {unified_path}")

    # 1. Load unified dataset
    logger.info("Loading problems from %s", unified_path)
    with open(unified_path, "r", encoding="utf-8") as f:
        raw_problems = json.load(f)
    problems_by_id = {str(p["id"]): p for p in raw_problems if p.get("id")}
    logger.info("Loaded %d problems with IDs from unified dataset", len(problems_by_id))

    # 2. Compute or load embeddings
    embeddings, ids = compute_all_embeddings(
        problems=raw_problems,
        unified_json_path=unified_path,
        cache_dir=cache_dir,
        recompute=recompute_embeddings,
    )
    n_records = len(ids)

    # 3. Cluster embeddings
    c_params: dict[str, Any] = {
        "umap_dim": 64,
        "min_cluster_size": 10,
        "min_samples": 1,
        "metric": "cosine",
        "cluster_selection_method": "eom",
        "random_state": 42,
        "assign_noise_threshold": 0.75,
    }
    if cluster_params:
        c_params.update(cluster_params)

    labels = cluster_embeddings(embeddings=embeddings, ids=ids, **c_params)

    # 4. Summarize clusters
    pattern_summaries = summarize_clusters(
        labels=labels,
        ids=ids,
        problems_by_id=problems_by_id,
        embeddings=embeddings,
    )

    n_clusters = len(pattern_summaries)
    outliers = int(np.sum(labels == -1))
    outlier_rate = round(outliers / n_records, 4) if n_records > 0 else 0.0

    # 5. Persist to DB
    logger.info("Persisting %d patterns and member assignments to %s", n_clusters, db_path)
    migrate_db(db_path)
    write_patterns(pattern_summaries, db_path=db_path)

    # Prepare member assignments (confidence 1.0 for HDBSCAN cluster members)
    assignments = []
    assigned_at = datetime.now(timezone.utc).isoformat()
    for pattern in pattern_summaries:
        pat_id = pattern["pattern_id"]
        for mid in pattern["members"]:
            assignments.append((mid, pat_id, 1.0, assigned_at))

    write_assignments(assignments, db_path=db_path)
    logger.info("Persisted %d pattern assignments to database", len(assignments))

    # 6. Export data/output/patterns.json
    patterns_export = []
    for p in pattern_summaries:
        p_clean = dict(p)
        # remove numpy array for JSON export
        p_clean.pop("centroid", None)
        patterns_export.append(p_clean)

    export_payload = {
        "summary": {
            "n_records": n_records,
            "n_clusters": n_clusters,
            "outliers": outliers,
            "outlier_rate": outlier_rate,
            "clustered_records": n_records - outliers,
            "cluster_params": c_params,
            "generated_at": assigned_at,
        },
        "patterns": patterns_export,
    }

    out_json_path = Path(output_patterns_path)
    _atomic_write_json(out_json_path, export_payload)
    logger.info("Exported pattern summaries to %s", out_json_path)

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    metrics = {
        "n_records": n_records,
        "n_clusters": n_clusters,
        "outliers": outliers,
        "outlier_rate": outlier_rate,
        "elapsed_seconds": round(elapsed, 2),
        "patterns_path": str(out_json_path),
        "db_path": str(db_path),
    }

    print("\n" + "=" * 50)
    print("Pattern Mining Pipeline Execution Complete")
    print(f"  Records processed: {n_records}")
    print(f"  Clusters discovered: {n_clusters}")
    print(f"  Outliers: {outliers} ({outlier_rate:.2%})")
    print(f"  Patterns file: {out_json_path}")
    print(f"  Database: {db_path}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print("=" * 50 + "\n")

    return metrics
