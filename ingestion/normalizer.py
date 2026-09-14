"""Normalizer.

Cleans raw text, canonicalizes difficulty and tag vocabulary, and extracts
examples into a structured list. Converts a `RawProblem` into a
`CanonicalRecord` (schema/fingerprints/license are filled in by later
stages, not here).
"""
from __future__ import annotations

import re
import uuid
from html import unescape

from .models import CanonicalRecord, Difficulty, Example, RawProblem

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TAG_RE = re.compile(r"<[^>]+>")

_DIFFICULTY_MAP = {
    "easy": Difficulty.EASY,
    "e": Difficulty.EASY,
    "1": Difficulty.EASY,
    "medium": Difficulty.MEDIUM,
    "med": Difficulty.MEDIUM,
    "m": Difficulty.MEDIUM,
    "2": Difficulty.MEDIUM,
    "hard": Difficulty.HARD,
    "h": Difficulty.HARD,
    "3": Difficulty.HARD,
    # Chinese difficulty labels used by doocs/leetcode for problem sets that
    # only ship a Chinese README (lcof/lcof2/lcp/lcs).
    "简单": Difficulty.EASY,
    "中等": Difficulty.MEDIUM,
    "困难": Difficulty.HARD,
}

# Canonical tag vocabulary mapping. Keys are normalized (lowercase,
# non-alnum stripped) aliases; values are the canonical tag name.
_TAG_ALIASES = {
    "array": "array",
    "arrays": "array",
    "hashmap": "hash-map",
    "hashtable": "hash-map",
    "hash-map": "hash-map",
    "dict": "hash-map",
    "string": "string",
    "strings": "string",
    "dp": "dynamic-programming",
    "dynamicprogramming": "dynamic-programming",
    "graph": "graph",
    "graphs": "graph",
    "tree": "tree",
    "trees": "tree",
    "binarytree": "binary-tree",
    "linkedlist": "linked-list",
    "twopointers": "two-pointers",
    "twopointer": "two-pointers",
    "slidingwindow": "sliding-window",
    "greedy": "greedy",
    "backtracking": "backtracking",
    "recursion": "recursion",
    "math": "math",
    "sorting": "sorting",
    "binarysearch": "binary-search",
    "stack": "stack",
    "queue": "queue",
    "heap": "heap",
    "priorityqueue": "heap",
    "bfs": "bfs",
    "dfs": "dfs",
    "bitmanipulation": "bit-manipulation",
    "trie": "trie",
    "unionfind": "union-find",
    "disjointset": "union-find",
}


def clean_text(text: str | None) -> str:
    """Strips HTML tags, unescapes entities, and normalizes whitespace."""
    if not text:
        return ""
    text = unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def canonicalize_difficulty(raw: str | None) -> Difficulty:
    if not raw:
        return Difficulty.UNKNOWN
    key = str(raw).strip().lower()
    return _DIFFICULTY_MAP.get(key, Difficulty.UNKNOWN)


def canonicalize_tag(raw: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", raw.strip().lower())
    return _TAG_ALIASES.get(key, raw.strip().lower().replace(" ", "-").replace("_", "-"))


def canonicalize_tags(raw_tags: list[str]) -> list[str]:
    seen: list[str] = []
    for tag in raw_tags:
        if not tag:
            continue
        canon = canonicalize_tag(tag)
        if canon and canon not in seen:
            seen.append(canon)
    return seen if seen else ["unknown"]


def extract_examples(examples_raw: list[dict]) -> list[Example]:
    examples: list[Example] = []
    for item in examples_raw:
        if not isinstance(item, dict):
            continue
        inp = item.get("input") or item.get("in") or item.get("input_data")
        out = item.get("output") or item.get("out") or item.get("expected") or item.get("expected_output")
        if inp is None or out is None:
            continue
        examples.append(
            Example(
                input=str(inp).strip(),
                output=str(out).strip(),
                explanation=(clean_text(item.get("explanation")) or None),
            )
        )
    return examples


def normalize(raw: RawProblem, ingest_id: str) -> CanonicalRecord:
    """Converts a RawProblem into a CanonicalRecord with cleaned/canonical
    text fields. Schema inference, fingerprints, license flags, and
    validation are filled in by later pipeline stages.
    """
    from .models import IngestTrace, LicenseMeta  # local import avoids cycle risk

    title = clean_text(raw.title)
    description = clean_text(raw.description)
    difficulty = canonicalize_difficulty(raw.difficulty)
    tags = canonicalize_tags(raw.tags)
    examples = extract_examples(raw.examples_raw)
    constraints = [clean_text(c) for c in raw.constraints if clean_text(c)]

    record = CanonicalRecord(
        template_source_id=str(uuid.uuid4()),
        title=title,
        description=description,
        difficulty=difficulty,
        tags=tags,
        examples=examples,
        constraints=constraints,
        has_solution=bool(raw.solution_code),
        is_premium=raw.is_premium,
        # Only a real data gap, not just "this problem is paywalled on
        # leetcode.com": most premium problems still ship a full
        # description + community solution via doocs. Dedup can still
        # clear this later if another source contributes a solution.
        data_unavailable=raw.is_premium and not bool(raw.solution_code),
        source_list=[raw.source_name],
        ingest_trace=IngestTrace(
            parser=raw.parser,
            raw_file=raw.raw_file,
            raw_offset=raw.raw_offset,
            commit_hash=raw.commit_hash,
        ),
        license_meta=LicenseMeta(
            source=raw.source_name,
            source_url=raw.source_url,
            license=raw.license,
            ingest_id=ingest_id,
        ),
    )

    if raw.solution_code:
        from .models import CanonicalSolution

        record.canonical_solution = CanonicalSolution(
            languages=[raw.solution_language] if raw.solution_language else [],
            code=raw.solution_code,
        )

    record.touch("normalized", detail=f"parser={raw.parser}")
    return record
