"""Shared canonical-record fixtures for transform/ tests. Deliberately
hand-built minimal dicts (not loaded from the real unified_dataset.json)
so these tests don't depend on the real dataset being present/unchanged.
"""
from __future__ import annotations

TWO_SUM_RECORD = {
    "id": "canonical-two-sum-001",
    "title": "Two Sum",
    "description": "Given an array of integers nums and an integer target, return indices of the two numbers that add up to target.",
    "difficulty": "easy",
    "tags": ["array", "hash-map"],
    "input_schema": {
        "type": "object",
        "properties": {"nums": {"type": "array<integer>"}, "target": {"type": "integer"}},
        "required": ["nums", "target"],
    },
    "output_schema": {"type": "array<integer>"},
    "examples": [
        {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"},
        {"input": "nums = [3,2,4], target = 6", "output": "[1,2]"},
    ],
    "constraints": ["2 <= nums.length <= 10^4"],
    "has_solution": True,
    "canonical_solution": {
        "languages": ["python"],
        "code_hashes": [],
        "ast_signature": "deadbeef",
        "code": (
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        d = {}\n"
            "        for i, x in enumerate(nums):\n"
            "            if (y := target - x) in d:\n"
            "                return [d[y], i]\n"
            "            d[x] = i\n"
        ),
    },
    "source_list": ["doocs-leetcode"],
    "license_meta": {"license": "cc-by-sa-4.0"},
    "ingest_trace": {"commit_hash": "abc123"},
}

MERGE_INTERVALS_RECORD = {
    "id": "canonical-merge-intervals-001",
    "title": "Merge Intervals",
    "description": "Given an array of intervals, merge all overlapping intervals.",
    "difficulty": "medium",
    "tags": ["array", "sorting"],
    "input_schema": {
        "type": "object",
        "properties": {"intervals": {"type": "array<array<integer>>"}},
        "required": ["intervals"],
    },
    "output_schema": {"type": "array<array<integer>>"},
    "examples": [
        {"input": "intervals = [[1,3],[2,6],[8,10],[15,18]]", "output": "[[1,6],[8,10],[15,18]]"},
    ],
    "constraints": [],
    "has_solution": True,
    "canonical_solution": {
        "languages": ["python"],
        "code_hashes": [],
        "ast_signature": "feedbead",
        "code": (
            "class Solution:\n"
            "    def merge(self, intervals):\n"
            "        intervals.sort()\n"
            "        merged = []\n"
            "        for start, end in intervals:\n"
            "            if merged and start <= merged[-1][1]:\n"
            "                merged[-1][1] = max(merged[-1][1], end)\n"
            "            else:\n"
            "                merged.append([start, end])\n"
            "        return merged\n"
        ),
    },
    "source_list": ["doocs-leetcode"],
    "license_meta": {"license": "cc-by-sa-4.0"},
    "ingest_trace": {"commit_hash": "abc123"},
}

NO_SOLUTION_RECORD = {
    "id": "canonical-no-solution-001",
    "title": "Locked Problem",
    "description": "A premium problem with no community solution.",
    "difficulty": "hard",
    "tags": ["array"],
    "input_schema": {},
    "output_schema": {},
    "examples": [{"input": "n = 1", "output": "1"}],
    "constraints": [],
    "has_solution": False,
    "canonical_solution": None,
    "source_list": ["doocs-leetcode"],
    "license_meta": {"license": "cc-by-sa-4.0"},
    "ingest_trace": {"commit_hash": None},
}

GOOD_TWO_SUM_REWRITE = {
    "title": "Transaction Pair Match",
    "description": "Find two transactions in order that sum to a target reimbursement.",
    "input_format": "transactions: list of int, target: int",
    "output_format": "list of two indices",
    "input_schema": {
        "type": "object",
        "properties": {"transactions": {"type": "array<integer>"}, "target": {"type": "integer"}},
    },
    "output_schema": {"type": "array<integer>"},
    "constraints": ["2 <= transactions.length <= 10000"],
    "tie_breaker": "prefer smallest first index",
    "notes_for_variant_generator": {"n_range": [2, 20], "value_range": [-100, 100], "special_flags": []},
    "sample_public_tests": [{"input": "transactions=[2,7,11,15],target=9", "output": "[0,1]"}],
    "sample_hidden_tests_design": "duplicates, negatives, large n",
    "transform_seed": "fixed-test-seed",
    "exemplar_core_logic": "hash map lookup of complements",
}
