"""Fingerprinting.

Computes:
  * `content_hash` — sha256 over normalized title+description, for exact
    dedup and idempotency checks.
  * `nl_embedding` — a lightweight TF-IDF-style bag-of-words vector over
    the corpus, used as a stand-in for a real sentence-transformer
    embedding (SBERT). The interface (`embed_corpus`) is the seam where a
    real embedding model would be swapped in for production use — nothing
    downstream needs to change since dedup/search only consume the
    resulting vectors.
  * `ast_signature` — a normalized signature of solution code (whitespace
    stripped, identifiers not resolved — a real implementation would use
    tree-sitter per spec section 11) used for code-level near-duplicate
    detection.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .models import CanonicalRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_hash(record: CanonicalRecord) -> str:
    basis = f"{record.title.strip().lower()}\n{record.description.strip().lower()}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def code_ast_signature(code: str) -> str:
    """Normalizes code text (strip comments-ish noise, then collapse
    whitespace) and hashes it. This is a coarse stand-in for a real
    tree-sitter AST n-gram signature — good enough to catch
    verbatim/near-verbatim copies, not semantic-equivalent rewrites.

    Comments MUST be stripped per-line, before whitespace collapse. A
    single `re.sub(r"#.*", "", ...)` applied after collapsing whitespace
    into one line has no newline left to anchor against, so it matches
    from the first `#` anywhere in the code to the literal end of the
    string — silently deleting the entire solution body whenever it
    starts with a leading comment block (e.g. doocs's near-universal
    "# Definition for singly-linked list..." header on linked-list
    problems). That bug previously went unnoticed because this function
    was never actually exercised on real code in the production pipeline
    (the `raw_solution_code` side-channel dict `apply_fingerprints` used
    to depend on was never populated by `pipeline.py`).
    """
    if not code:
        return ""
    no_comments = "\n".join(re.sub(r"#.*$", "", line) for line in code.splitlines())
    normalized = re.sub(r"\s+", " ", no_comments.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Corpus:
    """Fits a simple TF-IDF vector space over a batch of records' text so
    embeddings are comparable via cosine similarity. This intentionally
    avoids a heavyweight ML dependency; swap `embed_corpus` for a real
    SBERT model without touching callers.
    """

    def __init__(self, documents: list[str]):
        self.documents = documents
        self.doc_tokens = [tokenize(d) for d in documents]
        self.df: Counter = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1
        self.n_docs = max(len(documents), 1)
        self.vocab = sorted(self.df.keys())
        self.vocab_index = {term: i for i, term in enumerate(self.vocab)}

    def vector(self, tokens: list[str]) -> list[float]:
        if not self.vocab:
            return []
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for term, count in tf.items():
            idx = self.vocab_index.get(term)
            if idx is None:
                continue
            idf = math.log((self.n_docs + 1) / (self.df[term] + 1)) + 1.0
            vec[idx] = count * idf
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def vectors(self) -> list[list[float]]:
        return [self.vector(tokens) for tokens in self.doc_tokens]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))  # vectors are already L2-normalized


def embed_corpus(records: list[CanonicalRecord]) -> list[list[float]]:
    """Fits a TF-IDF space over all records' (title + description) and
    returns one vector per record, in order. This is the seam to replace
    with a real sentence-transformer model call.
    """
    docs = [f"{r.title} {r.description}" for r in records]
    corpus = Corpus(docs)
    return corpus.vectors()


def apply_fingerprints(records: list[CanonicalRecord], raw_solution_code: dict[str, str] | None = None) -> list[CanonicalRecord]:
    """Fills in fingerprints for a batch of records.

    Solution code for AST signature/hash purposes is read directly off
    `record.canonical_solution.code` (persisted by the normalizer) now
    that adapters' solution text is no longer discarded after parsing.
    `raw_solution_code` remains as an optional id->code override for
    callers that still want to pass code out-of-band; it takes
    precedence over the record's own `canonical_solution.code` when both
    are present.
    """
    raw_solution_code = raw_solution_code or {}
    vectors = embed_corpus(records)

    for record, vector in zip(records, vectors):
        record.fingerprints.content_hash = content_hash(record)
        record.fingerprints.nl_embedding = vector
        record.fingerprints.nl_embedding_id = f"nl:{record.id}"

        code = raw_solution_code.get(record.id) or (
            record.canonical_solution.code if record.canonical_solution else None
        )
        if code and record.canonical_solution is not None:
            sig = code_ast_signature(code)
            record.canonical_solution.ast_signature = sig
            record.canonical_solution.code_hashes = [hashlib.sha256(code.encode("utf-8")).hexdigest()]
            record.fingerprints.code_embedding_id = f"code:{record.id}"

        record.touch("fingerprinted")

    return records
