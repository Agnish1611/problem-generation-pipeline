import hashlib

from ingestion.fingerprint import (
    apply_fingerprints,
    code_ast_signature,
    content_hash,
    cosine_similarity,
    embed_corpus,
)
from ingestion.models import CanonicalRecord


def _record(title, description):
    return CanonicalRecord(title=title, description=description)


def test_content_hash_stable_and_case_insensitive():
    r1 = _record("Two Sum", "Some description")
    r2 = _record("two sum", "some description")
    assert content_hash(r1) == content_hash(r2)


def test_content_hash_differs_for_different_text():
    r1 = _record("Two Sum", "desc a")
    r2 = _record("Three Sum", "desc b")
    assert content_hash(r1) != content_hash(r2)


def test_embed_corpus_similar_docs_have_higher_similarity():
    records = [
        _record("Two Sum", "Given an array find two numbers that add up to a target value"),
        _record("Two Sum II", "Given a sorted array find two numbers that add up to a target value"),
        _record("Merge Intervals", "Given a list of intervals merge all overlapping intervals together"),
    ]
    vectors = embed_corpus(records)
    sim_related = cosine_similarity(vectors[0], vectors[1])
    sim_unrelated = cosine_similarity(vectors[0], vectors[2])
    assert sim_related > sim_unrelated


def test_code_ast_signature_matches_whitespace_variants():
    code_a = "def f(x):\n    return x + 1\n"
    code_b = "def f(x):   return x + 1"
    assert code_ast_signature(code_a) == code_ast_signature(code_b)


def test_code_ast_signature_strips_comments_per_line_not_to_end_of_string():
    # Regression test: a naive `re.sub(r"#.*", "", code)` applied AFTER
    # collapsing all whitespace/newlines onto one line has no newline
    # left to anchor against, so it deletes everything from the first
    # "#" character to the literal end of the string. That's exactly the
    # doocs/leetcode shape for every linked-list/tree problem, which
    # opens with a multi-line "# Definition for singly-linked list..."
    # comment block before any real code — the bug silently reduced
    # every such solution's signature to hash(""), making all of them
    # collide with each other and with truly-empty code.
    code_with_leading_comment_block = (
        "# Definition for singly-linked list.\n"
        "# class ListNode:\n"
        "#     def __init__(self, val=0, next=None):\n"
        "#         self.val = val\n"
        "class Solution:\n"
        "    def solve(self, head):\n"
        "        return head\n"
    )
    sig = code_ast_signature(code_with_leading_comment_block)
    assert sig != code_ast_signature("")  # must not collapse to the empty-code signature
    assert sig != hashlib.sha256(b"").hexdigest()


def test_code_ast_signature_differs_for_genuinely_different_code_with_similar_comment_headers():
    # Two different solutions that happen to share the same doocs-style
    # leading comment block must NOT hash to the same signature.
    header = "# Definition for singly-linked list.\n# class ListNode: pass\n"
    code_a = header + "class Solution:\n    def partition(self, head, x):\n        return head\n"
    code_b = header + "class Solution:\n    def isPalindrome(self, head):\n        return True\n"
    assert code_ast_signature(code_a) != code_ast_signature(code_b)


def test_apply_fingerprints_sets_content_hash_and_embedding():
    records = [_record("A", "Description one two three"), _record("B", "Description four five six")]
    apply_fingerprints(records)
    for r in records:
        assert r.fingerprints.content_hash
        assert r.fingerprints.nl_embedding is not None
        assert r.fingerprints.nl_embedding_id == f"nl:{r.id}"


def test_apply_fingerprints_reads_code_from_canonical_solution():
    from ingestion.models import CanonicalSolution

    record = _record("Two Sum", "Given an array find two numbers that sum to a target")
    record.canonical_solution = CanonicalSolution(languages=["python"], code="def f(x):\n    return x + 1\n")

    apply_fingerprints([record])

    assert record.canonical_solution.ast_signature == code_ast_signature("def f(x):\n    return x + 1\n")
    assert record.canonical_solution.code_hashes  # populated, not left empty
    assert record.fingerprints.code_embedding_id == f"code:{record.id}"
    # the raw code text itself must still be there, untouched
    assert record.canonical_solution.code == "def f(x):\n    return x + 1\n"


def test_apply_fingerprints_raw_solution_code_override_takes_precedence():
    from ingestion.models import CanonicalSolution

    record = _record("Two Sum", "Given an array find two numbers that sum to a target")
    record.canonical_solution = CanonicalSolution(languages=["python"], code="def old(): pass")

    override_code = "def new(): pass"
    apply_fingerprints([record], raw_solution_code={record.id: override_code})

    assert record.canonical_solution.ast_signature == code_ast_signature(override_code)
