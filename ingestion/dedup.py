"""De-duplication.

Implements the staged approach from the spec (section 6):

  1. Cheap exact-key match on normalized title.
  2. Token-overlap (Jaccard) on title + first N tokens of description.
  3. Embedding cosine similarity on NL fingerprints.
  4. Code-AST signature match for canonical solutions.

Matches consolidate into a single canonical record: the earliest-created
record survives as canonical, absorbing `source_list`, `aliases`, and
`tags` from the duplicate. The duplicate record is marked
`deduplicated_into` and dropped from the final output, but its provenance
is preserved on the canonical record's history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .fingerprint import cosine_similarity, tokenize
from .models import CanonicalRecord, Difficulty

TITLE_JACCARD_THRESHOLD = 0.8
DESCRIPTION_JACCARD_THRESHOLD = 0.6
EMBEDDING_SIMILARITY_THRESHOLD = 0.90
AST_SIGNATURE_MATCH = True  # exact match required for code-level dedup

DESCRIPTION_TOKEN_WINDOW = 50

_STOPWORDS = {
    "a", "an", "the", "of", "in", "to", "for", "with", "on", "at", "from",
    "by", "and", "or", "is", "it", "ii", "iii", "iv", "v", "i", "find",
    "number", "maximum", "minimum", "count",
}

# LeetCode-style sequel suffixes ("Two Sum" -> "Two Sum II" -> "Two Sum
# III", or "... Part I"/"... Part 2"). These are DIFFERENT problems that
# deliberately reuse the parent's setup boilerplate almost verbatim, which
# makes their descriptions look like near-duplicates to both the token
# Jaccard and embedding-similarity stages. Titles differing only by one of
# these suffixes must never be merged, regardless of description
# similarity.
_SEQUEL_SUFFIX_RE = re.compile(
    r"\s+(?:(?:I{1,3}|IV|V|VI)|Part\s+\d+|Part\s+(?:One|Two|Three|Four|Five))\s*$",
    re.IGNORECASE,
)


def _strip_sequel_suffix(title: str) -> str:
    return _SEQUEL_SUFFIX_RE.sub("", title).strip().lower()


def _is_sequel_pair(title_a: str, title_b: str) -> bool:
    """True if the two titles are the same base problem name but with
    different (or one missing) sequel-numbering suffixes — e.g.
    "Minimum Partition Score" vs "Minimum Partition Score II". Such pairs
    must be treated as distinct problems, never merged as duplicates.
    """
    if title_a.strip().lower() == title_b.strip().lower():
        return False  # identical titles are handled by the exact-key stage
    base_a = _strip_sequel_suffix(title_a)
    base_b = _strip_sequel_suffix(title_b)
    return bool(base_a) and base_a == base_b


def _norm_title_key(title: str) -> str:
    return " ".join(tokenize(title))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class DedupMatch:
    canonical: CanonicalRecord
    duplicate: CanonicalRecord
    stage: str
    score: float


def _candidate_pairs(records: list[CanonicalRecord]) -> list[tuple[int, int]]:
    """Only compare records sharing at least one informative title token,
    filtering common stopwords to keep pair counts manageable on large corpora.
    """
    buckets: dict[str, list[int]] = {}
    token_sets: list[set[str]] = []
    for i, r in enumerate(records):
        toks = set(tokenize(r.title))
        token_sets.append(toks)
        for tok in (toks - _STOPWORDS):
            buckets.setdefault(tok, []).append(i)

    pairs: set[tuple[int, int]] = set()
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.add((min(idxs[a], idxs[b]), max(idxs[a], idxs[b])))

    pruned: list[tuple[int, int]] = []
    for i, j in pairs:
        sim = _jaccard(token_sets[i], token_sets[j])
        if sim >= 0.25:
            pruned.append((i, j))
    return sorted(pruned)


def find_duplicates(records: list[CanonicalRecord]) -> list[DedupMatch]:
    matches: list[DedupMatch] = []
    matched_duplicate_idx: set[int] = set()

    # Stage 1: Exact normalized title match via O(N) hash grouping.
    title_buckets: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        key = _norm_title_key(r.title)
        if key:
            title_buckets.setdefault(key, []).append(i)

    for key, idxs in title_buckets.items():
        if len(idxs) < 2:
            continue
        # Canonical survivor = earliest created_at (stable tie-break by id).
        survivor_idx = min(idxs, key=lambda idx: (records[idx].created_at, records[idx].id))
        for dup_idx in idxs:
            if dup_idx == survivor_idx:
                continue
            canonical = records[survivor_idx]
            duplicate = records[dup_idx]
            matches.append(DedupMatch(canonical, duplicate, "title_key", 1.0))
            matched_duplicate_idx.add(dup_idx)

    # Near-duplicate stages for remaining candidate pairs.
    for i, j in _candidate_pairs(records):
        if i in matched_duplicate_idx or j in matched_duplicate_idx:
            continue
        rec_a, rec_b = records[i], records[j]
        if rec_a.deduplicated_into or rec_b.deduplicated_into:
            continue
        if _is_sequel_pair(rec_a.title, rec_b.title):
            # e.g. "Minimum Partition Score" vs "... II" — distinct
            # problems that share near-identical setup boilerplate, so
            # skip every near-duplicate stage for this pair entirely.
            continue

        canonical, duplicate = (
            (rec_a, rec_b)
            if (rec_a.created_at, rec_a.id) <= (rec_b.created_at, rec_b.id)
            else (rec_b, rec_a)
        )
        dup_idx = j if canonical is rec_a else i

        # Stage 2: token Jaccard on title + description window.
        title_sim = _jaccard(set(tokenize(canonical.title)), set(tokenize(duplicate.title)))
        desc_a = set(tokenize(canonical.description)[:DESCRIPTION_TOKEN_WINDOW])
        desc_b = set(tokenize(duplicate.description)[:DESCRIPTION_TOKEN_WINDOW])
        desc_sim = _jaccard(desc_a, desc_b)
        if title_sim >= TITLE_JACCARD_THRESHOLD and desc_sim >= DESCRIPTION_JACCARD_THRESHOLD:
            matches.append(DedupMatch(canonical, duplicate, "token_jaccard", (title_sim + desc_sim) / 2))
            matched_duplicate_idx.add(dup_idx)
            continue

        # Stage 3: embedding cosine similarity.
        emb_a = canonical.fingerprints.nl_embedding
        emb_b = duplicate.fingerprints.nl_embedding
        if emb_a and emb_b:
            sim = cosine_similarity(emb_a, emb_b)
            if sim >= EMBEDDING_SIMILARITY_THRESHOLD:
                matches.append(DedupMatch(canonical, duplicate, "embedding", sim))
                matched_duplicate_idx.add(dup_idx)
                continue

        # Stage 4: code AST signature exact match (near-copy solutions).
        sig_a = canonical.canonical_solution.ast_signature if canonical.canonical_solution else None
        sig_b = duplicate.canonical_solution.ast_signature if duplicate.canonical_solution else None
        if sig_a and sig_b and sig_a == sig_b:
            matches.append(DedupMatch(canonical, duplicate, "code_ast", 1.0))
            matched_duplicate_idx.add(dup_idx)
            continue

    return matches


def consolidate(records: list[CanonicalRecord]) -> list[CanonicalRecord]:
    """Applies `find_duplicates` and merges duplicates into their
    canonical record in place. Returns the surviving (non-duplicate)
    records; duplicates remain in the input list with
    `deduplicated_into` set, so callers can still persist them for audit
    purposes if desired.
    """
    matches = find_duplicates(records)

    for match in matches:
        canonical, duplicate = match.canonical, match.duplicate

        for src in duplicate.source_list:
            if src not in canonical.source_list:
                canonical.source_list.append(src)

        if duplicate.title != canonical.title and duplicate.title not in canonical.aliases:
            canonical.aliases.append(duplicate.title)

        for tag in duplicate.tags:
            if tag not in canonical.tags and tag != "unknown":
                canonical.tags.append(tag)

        if not canonical.has_solution and duplicate.has_solution:
            canonical.has_solution = True
            canonical.canonical_solution = duplicate.canonical_solution

        # is_premium is a factual property of the problem (paywalled on
        # leetcode.com) — keep it if either side flagged it. data_unavailable
        # tracks whether we actually lack usable data for it; once any
        # source contributes a solution, that gap is closed regardless of
        # which side ends up "canonical".
        canonical.is_premium = canonical.is_premium or duplicate.is_premium
        canonical.data_unavailable = canonical.is_premium and not canonical.has_solution

        if not canonical.description and duplicate.description:
            canonical.description = duplicate.description

        if not canonical.examples and duplicate.examples:
            canonical.examples = duplicate.examples
            canonical.input_schema = duplicate.input_schema
            canonical.output_schema = duplicate.output_schema
            canonical.schema_inference_confidence = duplicate.schema_inference_confidence

        if not canonical.constraints and duplicate.constraints:
            canonical.constraints = duplicate.constraints

        if canonical.difficulty == Difficulty.UNKNOWN and duplicate.difficulty != Difficulty.UNKNOWN:
            canonical.difficulty = duplicate.difficulty

        duplicate.deduplicated_into = canonical.id
        duplicate.touch("deduplicated", detail=f"stage={match.stage} score={match.score:.3f} into={canonical.id}")
        canonical.bump_version(
            "consolidated_duplicate",
            detail=f"absorbed {duplicate.id} via {match.stage} (score={match.score:.3f})",
        )

    return [r for r in records if not r.deduplicated_into]
