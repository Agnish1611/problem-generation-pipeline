from sandbox.runner import (
    RunOutcome,
    detect_entry_point,
    parse_kwargs,
    run_solution_on_examples,
    run_test_cases,
)

TWO_SUM_CODE = """
class Solution:
    def twoSum(self, nums, target):
        d = {}
        for i, x in enumerate(nums):
            if (y := target - x) in d:
                return [d[y], i]
            d[x] = i
"""

TWO_SUM_EXAMPLES = [
    {"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]"},
    {"input": "nums = [3,2,4], target = 6", "output": "[1,2]"},
]


def test_detect_entry_point_finds_solution_method():
    entry = detect_entry_point(TWO_SUM_CODE)
    assert entry is not None
    assert entry.kind == "method"
    assert entry.name == "twoSum"


def test_detect_entry_point_finds_top_level_function():
    entry = detect_entry_point("def add(a, b):\n    return a + b\n")
    assert entry is not None
    assert entry.kind == "function"
    assert entry.name == "add"


def test_detect_entry_point_skips_private_methods():
    code = """
class Solution:
    def _helper(self):
        pass
    def solve(self, n):
        return n
"""
    entry = detect_entry_point(code)
    assert entry is not None
    assert entry.name == "solve"


def test_detect_entry_point_returns_none_for_syntax_error():
    assert detect_entry_point("def broken(:\n") is None


def test_detect_entry_point_returns_none_when_nothing_callable():
    assert detect_entry_point("x = 5\n") is None


def test_parse_kwargs_handles_leetcode_style_multi_arg_input():
    args = parse_kwargs("nums = [2,7,11,15], target = 9")
    assert args == {"nums": [2, 7, 11, 15], "target": 9}


def test_parse_kwargs_handles_nested_brackets_without_splitting_incorrectly():
    args = parse_kwargs("intervals = [[1,3],[2,6]], k = 2")
    assert args == {"intervals": [[1, 3], [2, 6]], "k": 2}


def test_run_test_cases_all_pass_for_correct_solution():
    result = run_test_cases(TWO_SUM_CODE, TWO_SUM_EXAMPLES)
    assert result.outcome == RunOutcome.OK
    assert result.all_passed is True
    assert len(result.cases) == 2
    assert all(c.passed for c in result.cases)


def test_run_test_cases_reports_no_examples():
    result = run_test_cases(TWO_SUM_CODE, [])
    assert result.outcome == RunOutcome.NO_EXAMPLES
    assert result.all_passed is False


def test_run_test_cases_reports_no_entry_point():
    result = run_test_cases("x = 1\n", TWO_SUM_EXAMPLES)
    assert result.outcome == RunOutcome.NO_ENTRY_POINT
    assert result.all_passed is False


def test_run_test_cases_catches_wrong_output():
    wrong_code = """
class Solution:
    def twoSum(self, nums, target):
        return [999, 999]
"""
    result = run_test_cases(wrong_code, TWO_SUM_EXAMPLES)
    assert result.outcome == RunOutcome.OK
    assert result.all_passed is False
    assert all(not c.passed for c in result.cases)
    assert result.cases[0].actual == [999, 999]


def test_run_test_cases_catches_runtime_exception():
    buggy_code = """
class Solution:
    def twoSum(self, nums, target):
        return 1 / 0
"""
    result = run_test_cases(buggy_code, TWO_SUM_EXAMPLES)
    assert result.outcome == RunOutcome.OK
    assert result.cases[0].passed is False
    assert result.cases[0].error is not None
    assert "ZeroDivisionError" in result.cases[0].error
    assert result.cases[0].timed_out is False


def test_run_test_cases_catches_infinite_loop_via_timeout():
    infinite_loop_code = """
class Solution:
    def twoSum(self, nums, target):
        while True:
            pass
"""
    result = run_test_cases(infinite_loop_code, TWO_SUM_EXAMPLES[:1], timeout_seconds=1.0)
    assert result.outcome == RunOutcome.OK
    assert result.cases[0].timed_out is True
    assert result.cases[0].passed is False


def test_run_test_cases_handles_tuple_vs_list_return_value():
    # A solution returning a tuple should still compare equal to a
    # JSON-decoded list expected value (see _canonicalize).
    tuple_code = """
class Solution:
    def twoSum(self, nums, target):
        d = {}
        for i, x in enumerate(nums):
            if (y := target - x) in d:
                return (d[y], i)
            d[x] = i
"""
    result = run_test_cases(tuple_code, TWO_SUM_EXAMPLES)
    assert result.all_passed is True


def test_run_solution_on_examples_reports_unsupported_language():
    record = {
        "canonical_solution": {"languages": ["java"], "code": "class Solution {}"},
        "examples": TWO_SUM_EXAMPLES,
    }
    result = run_solution_on_examples(record)
    assert result.outcome == RunOutcome.UNSUPPORTED_LANGUAGE


def test_run_solution_on_examples_reports_missing_code():
    record = {
        "canonical_solution": {"languages": ["python"], "code": None},
        "examples": TWO_SUM_EXAMPLES,
    }
    result = run_solution_on_examples(record)
    assert result.outcome == RunOutcome.UNSUPPORTED_LANGUAGE


def test_run_solution_on_examples_runs_python_solution_end_to_end():
    record = {
        "canonical_solution": {"languages": ["python"], "code": TWO_SUM_CODE},
        "examples": TWO_SUM_EXAMPLES,
    }
    result = run_solution_on_examples(record)
    assert result.outcome == RunOutcome.OK
    assert result.all_passed is True


def test_preamble_provides_itertools_count_without_colliding_with_counter():
    # Regression test: itertools.count was previously aliased to
    # "itercount" to avoid a (nonexistent) collision with
    # collections.Counter — different names, case-sensitive, no actual
    # collision — which broke solutions that use `count` directly (a
    # real doocs pattern: `for x in count(n + 1): ...`).
    code = """
class Solution:
    def solve(self, n):
        for x in count(n + 1):
            return x
"""
    result = run_test_cases(code, [{"input": "n = 5", "output": "6"}])
    assert result.all_passed is True


def test_preamble_provides_heapq_and_bisect_bare_names():
    # Regression test: doocs solutions commonly call heappush/heappop/
    # heapify/nlargest and bisect_left/bisect_right as bare names (not
    # via the `heapq.`/`bisect.` module prefix), relying on LeetCode's
    # auto-injected imports.
    heap_code = """
class Solution:
    def solve(self, nums):
        h = []
        for x in nums:
            heappush(h, x)
        return heappop(h)
"""
    result = run_test_cases(heap_code, [{"input": "nums = [3,1,2]", "output": "1"}])
    assert result.all_passed is True

    bisect_code = """
class Solution:
    def solve(self, nums, target):
        return bisect_left(nums, target)
"""
    result = run_test_cases(bisect_code, [{"input": "nums = [1,3,5], target = 3", "output": "1"}])
    assert result.all_passed is True


def test_preamble_provides_random_bare_names():
    # doocs solutions occasionally use randint/choice/shuffle directly
    # (e.g. reservoir-sampling / random-pick problems).
    code = """
class Solution:
    def solve(self, lo, hi):
        x = randint(lo, hi)
        return lo <= x <= hi
"""
    result = run_test_cases(code, [{"input": "lo = 1, hi = 10", "output": "true"}])
    assert result.all_passed is True
