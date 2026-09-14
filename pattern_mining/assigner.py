"""Incremental pattern assignment for newly ingested problems.

Assigns unseen problems to algorithmic patterns based on cosine similarity
to pattern centroids without requiring a full re-clustering run. Uses running
updates for centroids and persists assignments directly to SQLite.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embeddings import compose_text, get_sentence_transformer, get_text_embedding
from .storage import (
    get_assigned_content_hashes,
    get_assigned_problem_ids,
    migrate_db,
    read_patterns,
    update_pattern_centroid,
    write_assignments,
    write_patterns,
)

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.80


def _extract_next_pattern_index(patterns: List[dict[str, Any]]) -> int:
    """Find the next integer suffix for pattern_{index}."""
    max_idx = -1
    for p in patterns:
        pid = p.get("pattern_id", "")
        match = re.search(r"pattern_(\d+)$", pid)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        return vec / norm
    return vec


def _get_problem_content_hash(problem: dict[str, Any]) -> Optional[str]:
    fp = problem.get("fingerprints")
    if isinstance(fp, dict):
        return fp.get("content_hash")
    return problem.get("content_hash")


class CentroidTracker:
    """Maintains in-memory centroid vectors and metadata for fast similarity lookups."""

    def __init__(self, patterns: List[dict[str, Any]]):
        self.pattern_ids: List[str] = []
        self.centroids: List[np.ndarray] = []
        self.sizes: Dict[str, int] = {}
        self.meta: Dict[str, dict[str, Any]] = {}
        self.next_idx = _extract_next_pattern_index(patterns)

        for p in patterns:
            pid = p["pattern_id"]
            raw_c = p.get("centroid")
            if raw_c is None:
                raw_c = p.get("centroid_embedding")
            if raw_c is not None:
                c_arr = np.asarray(raw_c, dtype=np.float32)
                self.pattern_ids.append(pid)
                self.centroids.append(c_arr)
                self.sizes[pid] = int(p.get("size", p.get("count", 1)))
                self.meta[pid] = p

    @property
    def count(self) -> int:
        return len(self.pattern_ids)

    def get_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return (C_matrix, pattern_ids) where rows of C_matrix are normalized centroids."""
        if not self.centroids:
            return np.empty((0, 384), dtype=np.float32), []
        stacked = np.stack(self.centroids, axis=0)
        # Normalize rows for cosine similarity
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return stacked / norms, self.pattern_ids

    def update_centroid(self, pattern_id: str, new_emb: np.ndarray) -> Tuple[np.ndarray, int]:
        """Apply running average formula: (old_centroid * old_size + new_emb) / (old_size + 1)"""
        idx = self.pattern_ids.index(pattern_id)
        old_centroid = self.centroids[idx]
        old_size = self.sizes[pattern_id]

        new_size = old_size + 1
        new_centroid = (old_centroid * old_size + new_emb) / new_size
        new_centroid = new_centroid.astype(np.float32)

        self.centroids[idx] = new_centroid
        self.sizes[pattern_id] = new_size
        return new_centroid, new_size

    def add_pattern(
        self,
        pattern_id: str,
        centroid: np.ndarray,
        meta: dict[str, Any],
    ) -> None:
        self.pattern_ids.append(pattern_id)
        self.centroids.append(centroid.astype(np.float32))
        self.sizes[pattern_id] = 1
        self.meta[pattern_id] = meta


def assign_new_problem(
    problem: dict[str, Any],
    db_path: str | Path,
    model: Optional[Any] = None,
    tracker: Optional[CentroidTracker] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[str, float]:
    """Assign a single problem to an existing pattern or create a new pattern.

    - Computes embedding for problem.
    - Computes cosine similarity vs all stored pattern centroids.
    - If best similarity >= threshold, assigns to pattern, updates centroid via running
      average: new_centroid = (old_centroid * old_size + new_emb) / (old_size + 1),
      increments size, and persists to DB.
    - Else creates a new pattern row with next pattern_id (e.g. pattern_{next_int}) and persists.
    """
    migrate_db(db_path)
    pid = str(problem.get("id") or "")
    if not pid:
        raise ValueError("Problem must have a valid 'id' for assignment.")

    if tracker is None:
        patterns = read_patterns(db_path)
        tracker = CentroidTracker(patterns)

    # 1. Compute embedding
    if model is None:
        model = get_sentence_transformer()
    assert model is not None
    emb = get_text_embedding(problem, model=model)

    # 2. Compute similarity against centroids
    assigned_pattern_id: str
    confidence: float
    pids: List[str] = []

    if tracker.count > 0:
        norm_matrix, pids = tracker.get_matrix()
        norm_emb = _normalize_vector(emb)
        sims = np.dot(norm_matrix, norm_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
    else:
        best_sim = -1.0
        best_idx = -1

    assigned_at = datetime.now(timezone.utc).isoformat()

    if best_sim >= threshold and best_idx >= 0:
        assigned_pattern_id = pids[best_idx]
        confidence = best_sim

        # Update centroid running average
        new_centroid, new_size = tracker.update_centroid(assigned_pattern_id, emb)
        update_pattern_centroid(assigned_pattern_id, new_centroid, new_size, db_path)
        logger.info(
            "Assigned problem '%s' to pattern '%s' (similarity=%.4f, new size=%d)",
            pid,
            assigned_pattern_id,
            confidence,
            new_size,
        )
    else:
        # Create new pattern
        next_int = tracker.next_idx
        tracker.next_idx += 1
        assigned_pattern_id = f"pattern_{next_int}"
        confidence = 1.0

        title = str(problem.get("title", "New Algorithmic Pattern"))
        tags = problem.get("tags") or []
        diff = problem.get("difficulty", "unknown")

        new_pattern = {
            "pattern_id": assigned_pattern_id,
            "label": f"{tags[0]} / {title}" if tags else title,
            "size": 1,
            "representative": pid,
            "top_tags": tags[:6] if tags else [],
            "difficulty_json": {diff: 1},
            "centroid": emb,
            "created_at": assigned_at,
        }

        write_patterns([new_pattern], db_path)
        tracker.add_pattern(assigned_pattern_id, emb, new_pattern)
        logger.info(
            "Created new pattern '%s' for problem '%s' (best sim was %.4f < %.2f threshold)",
            assigned_pattern_id,
            pid,
            best_sim,
            threshold,
        )

    # Persist assignment
    write_assignments([(pid, assigned_pattern_id, confidence, assigned_at)], db_path)
    return assigned_pattern_id, confidence


def bulk_assign_new_records(
    list_of_problems: List[dict[str, Any]],
    db_path: str | Path,
    batch_size: int = 64,
    threshold: float = DEFAULT_THRESHOLD,
    model: Optional[Any] = None,
) -> List[dict[str, Any]]:
    """Incrementally assign a list of problems to patterns using vectorized batch processing.

    - Skips problems whose content_hash or id is already in pattern_assignments.
    - Loops through problems in batches, vectorizing embedding computation.
    - Updates centroids with running average without full re-clustering.
    """
    migrate_db(db_path)
    existing_assigned_ids = get_assigned_problem_ids(db_path)
    existing_assigned_hashes = get_assigned_content_hashes(db_path)

    # Filter out already assigned records
    unassigned_problems: List[dict[str, Any]] = []
    for p in list_of_problems:
        pid = p.get("id")
        ch = _get_problem_content_hash(p)

        if not pid or not p.get("title"):
            continue

        if str(pid) in existing_assigned_ids:
            continue

        if ch and ch in existing_assigned_hashes:
            continue

        unassigned_problems.append(p)

    if not unassigned_problems:
        logger.info("No unassigned problems found. Skipping incremental assignment.")
        return []

    logger.info("Found %d unassigned problems to process.", len(unassigned_problems))

    if model is None:
        model = get_sentence_transformer()
    assert model is not None

    # Load tracker from current DB state
    patterns = read_patterns(db_path)
    tracker = CentroidTracker(patterns)

    results: List[dict[str, Any]] = []

    # Process in batches
    for i in range(0, len(unassigned_problems), batch_size):
        batch = unassigned_problems[i : i + batch_size]
        texts = [compose_text(p) for p in batch]

        # Batch encode
        batch_embs = model.encode(
            texts,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        batch_embs = np.asarray(batch_embs, dtype=np.float32)

        for p, emb in zip(batch, batch_embs):
            pid = str(p["id"])
            assigned_at = datetime.now(timezone.utc).isoformat()
            pids: List[str] = []

            if tracker.count > 0:
                norm_matrix, pids = tracker.get_matrix()
                norm_emb = _normalize_vector(emb)
                sims = np.dot(norm_matrix, norm_emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
            else:
                best_sim = -1.0
                best_idx = -1

            if best_sim >= threshold and best_idx >= 0:
                assigned_pattern_id = pids[best_idx]
                confidence = best_sim
                new_centroid, new_size = tracker.update_centroid(assigned_pattern_id, emb)
                update_pattern_centroid(assigned_pattern_id, new_centroid, new_size, db_path)
            else:
                next_int = tracker.next_idx
                tracker.next_idx += 1
                assigned_pattern_id = f"pattern_{next_int}"
                confidence = 1.0

                title = str(p.get("title", "New Pattern"))
                tags = p.get("tags") or []
                diff = p.get("difficulty", "unknown")

                new_pattern = {
                    "pattern_id": assigned_pattern_id,
                    "label": f"{tags[0]} / {title}" if tags else title,
                    "size": 1,
                    "representative": pid,
                    "top_tags": tags[:6] if tags else [],
                    "difficulty_json": {diff: 1},
                    "centroid": emb,
                    "created_at": assigned_at,
                }
                write_patterns([new_pattern], db_path)
                tracker.add_pattern(assigned_pattern_id, emb, new_pattern)

            write_assignments([(pid, assigned_pattern_id, confidence, assigned_at)], db_path)
            results.append({
                "problem_id": pid,
                "pattern_id": assigned_pattern_id,
                "confidence": confidence,
                "assigned_at": assigned_at,
            })

    logger.info("Successfully assigned %d new problems.", len(results))
    return results
