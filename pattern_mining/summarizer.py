"""Cluster summarization and representative problem selection.

Generates concise algorithmic pattern summaries from HDBSCAN clusters,
including auto-generated labels, centroids, tag frequencies, and difficulty
distributions.
"""
from __future__ import annotations

import collections
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "to", "for", "with", "and", "or",
    "by", "at", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "this", "that", "how", "what", "which", "given", "find", "return",
    "determine", "calculate", "check", "using", "use",
}


def _extract_top_ngrams(titles: Sequence[str], top_k: int = 2) -> List[str]:
    """Extract top 1-gram and 2-gram keywords from cluster titles."""
    tokens_list = []
    bigrams_list = []

    for t in titles:
        # Extract alphanumeric words
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", t)]
        filtered = [w for w in words if w not in STOP_WORDS]
        tokens_list.extend(filtered)
        for i in range(len(filtered) - 1):
            bigrams_list.append(f"{filtered[i]} {filtered[i+1]}")

    bigram_counts = collections.Counter(bigrams_list)
    token_counts = collections.Counter(tokens_list)

    candidates: List[str] = []
    # Prefer repeated bigrams first
    for bg, c in bigram_counts.most_common(top_k):
        if c >= 2:
            candidates.append(bg)

    # Fill with unigrams
    for tok, _ in token_counts.most_common(top_k + 2):
        if tok not in " ".join(candidates) and len(candidates) < top_k:
            candidates.append(tok)

    return candidates


def generate_cluster_label(top_tags: List[str], titles: Sequence[str]) -> str:
    """Generate a short auto-label using top tags and top n-grams."""
    ngrams = _extract_top_ngrams(titles, top_k=2)

    parts: List[str] = []
    if top_tags:
        parts.extend(top_tags[:2])

    for ng in ngrams:
        if ng.lower() not in [p.lower() for p in parts]:
            parts.append(ng)
        if len(parts) >= 3:
            break

    if not parts:
        return "algorithmic-pattern"

    return " / ".join(parts[:3])


def summarize_cluster(
    cluster_label: int,
    member_ids: List[str],
    problems_by_id: Dict[str, dict[str, Any]],
    embeddings: np.ndarray,
    ids_index_map: Dict[str, int],
) -> dict[str, Any]:
    """Summarize a single cluster into an algorithmic pattern.

    Returns dict containing:
      pattern_id, label, count, representative, members, top_tags,
      difficulty_distribution, centroid_embedding, created_at
    """
    valid_members = [pid for pid in member_ids if pid in ids_index_map]
    if not valid_members:
        raise ValueError(f"Cluster {cluster_label} has no valid member embeddings.")

    indices = [ids_index_map[pid] for pid in valid_members]
    member_embs = embeddings[indices]

    # Compute centroid
    centroid = np.mean(member_embs, axis=0).astype(np.float32)

    # Find representative (closest to centroid by cosine distance)
    # Cosine distance = 1 - (dot(a, b) / (norm(a)*norm(b)))
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 1e-12:
        norm_centroid = centroid / centroid_norm
    else:
        norm_centroid = centroid

    members_norms = np.linalg.norm(member_embs, axis=1, keepdims=True)
    members_norms[members_norms == 0] = 1.0
    norm_members = member_embs / members_norms

    cos_sim = np.dot(norm_members, norm_centroid)
    best_idx_in_members = int(np.argmax(cos_sim))
    representative_id = valid_members[best_idx_in_members]

    # Collect tags, difficulties, titles
    tag_counter: collections.Counter[str] = collections.Counter()
    diff_counter: collections.Counter[str] = collections.Counter()
    titles: List[str] = []

    for pid in valid_members:
        p = problems_by_id.get(pid, {})
        tags = p.get("tags") or []
        for t in tags:
            if isinstance(t, str) and t.strip():
                tag_counter[t.strip().lower()] += 1

        diff = p.get("difficulty")
        if isinstance(diff, str) and diff.strip():
            diff_counter[diff.strip().lower()] += 1
        else:
            diff_counter["unknown"] += 1

        title = p.get("title")
        if title:
            titles.append(str(title))

    top_tags = [tag for tag, _ in tag_counter.most_common(6)]
    label = generate_cluster_label(top_tags, titles)
    created_at = datetime.now(timezone.utc).isoformat()

    return {
        "pattern_id": f"pattern_{cluster_label}",
        "label": label,
        "size": len(valid_members),
        "representative": representative_id,
        "members": valid_members,
        "top_tags": top_tags,
        "difficulty_distribution": dict(diff_counter),
        "centroid": centroid,
        "centroid_embedding": centroid.tolist(),
        "created_at": created_at,
    }


def summarize_clusters(
    labels: np.ndarray,
    ids: List[str],
    problems_by_id: Dict[str, dict[str, Any]],
    embeddings: np.ndarray,
) -> List[dict[str, Any]]:
    """Summarize all non -1 clusters into pattern dictionaries."""
    ids_index_map = {pid: idx for idx, pid in enumerate(ids)}

    cluster_to_members: Dict[int, List[str]] = collections.defaultdict(list)
    for pid, lbl in zip(ids, labels):
        if lbl != -1:
            cluster_to_members[int(lbl)].append(pid)

    summaries: List[dict[str, Any]] = []
    # Sort cluster labels for determinism
    for cl in sorted(cluster_to_members.keys()):
        member_ids = cluster_to_members[cl]
        summary = summarize_cluster(
            cluster_label=cl,
            member_ids=member_ids,
            problems_by_id=problems_by_id,
            embeddings=embeddings,
            ids_index_map=ids_index_map,
        )
        summaries.append(summary)

    logger.info("Summarized %d patterns across %d clustered items", len(summaries), sum(len(m) for m in cluster_to_members.values()))
    return summaries
