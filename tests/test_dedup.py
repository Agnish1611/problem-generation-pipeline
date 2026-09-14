from ingestion.dedup import consolidate
from ingestion.fingerprint import apply_fingerprints
from ingestion.models import CanonicalRecord, CanonicalSolution


def _record(title, description, source="src-a"):
    return CanonicalRecord(title=title, description=description, source_list=[source])


def test_consolidate_merges_exact_title_duplicates():
    records = [
        _record("Two Sum", "Given an array of integers find two numbers that sum to target", source="src-a"),
        _record("two sum", "Given an array of integers find two numbers that sum to target value", source="src-b"),
    ]
    apply_fingerprints(records)
    survivors = consolidate(records)

    assert len(survivors) == 1
    survivor = survivors[0]
    assert set(survivor.source_list) == {"src-a", "src-b"}
    # The duplicate itself should be marked, not deleted from the input list.
    duplicate = [r for r in records if r.id != survivor.id][0]
    assert duplicate.deduplicated_into == survivor.id


def test_consolidate_keeps_distinct_records_separate():
    records = [
        _record("Two Sum", "Given an array find two numbers that sum to a target value exactly"),
        _record("Merge Intervals", "Given a list of intervals merge all overlapping ranges together"),
    ]
    apply_fingerprints(records)
    survivors = consolidate(records)

    assert len(survivors) == 2


def test_consolidate_does_not_merge_leetcode_sequel_problems():
    # "X" and "X II" are distinct LeetCode problems that deliberately
    # reuse the parent's setup boilerplate almost verbatim, which makes
    # them look like near-duplicates by token/embedding similarity. They
    # must never be merged.
    boilerplate = (
        "You are given an integer array nums and an integer k. Your task is to partition nums "
        "into exactly k subarrays and return an integer denoting the minimum possible score "
        "among all valid partitions. The score of a partition is the sum of the values."
    )
    records = [
        _record("Minimum Partition Score", boilerplate, source="doocs-leetcode"),
        _record("Minimum Partition Score II", boilerplate, source="doocs-leetcode"),
    ]
    apply_fingerprints(records)
    survivors = consolidate(records)

    assert len(survivors) == 2
    assert {r.title for r in survivors} == {"Minimum Partition Score", "Minimum Partition Score II"}


def test_consolidate_clears_data_unavailable_when_duplicate_has_solution():
    # Simulates a premium doocs record with no solution merging with a
    # near-duplicate from another source that does carry a solution.
    premium_no_solution = _record(
        "Swap Salary", "Given a table salary swap the sex column values m and f", source="doocs-leetcode"
    )
    premium_no_solution.is_premium = True
    premium_no_solution.data_unavailable = True

    other_source_with_solution = _record(
        "Swap Salary", "Given a table salary swap the sex column values m and f exactly", source="leetcode-kaggle"
    )
    other_source_with_solution.has_solution = True
    other_source_with_solution.canonical_solution = CanonicalSolution(languages=["sql"], code="UPDATE salary ...")

    records = [premium_no_solution, other_source_with_solution]
    apply_fingerprints(records)
    survivors = consolidate(records)

    assert len(survivors) == 1
    survivor = survivors[0]
    assert survivor.is_premium is True
    assert survivor.has_solution is True
    assert survivor.data_unavailable is False
    # The actual solution code text, not just the has_solution flag, must
    # carry over from the duplicate that supplied it.
    assert survivor.canonical_solution is not None
    assert survivor.canonical_solution.code == "UPDATE salary ..."


def test_consolidate_merges_via_embedding_similarity_for_near_duplicate_titles():
    records = [
        _record(
            "Two Sum",
            "Given an array of integers nums and an integer target return indices of the two numbers that add up to target",
        ),
        _record(
            "Two Sum Problem",
            "Given an array of integers nums and an integer target return the indices of the two numbers that add up to target",
        ),
    ]
    apply_fingerprints(records)
    survivors = consolidate(records)
    assert len(survivors) == 1
