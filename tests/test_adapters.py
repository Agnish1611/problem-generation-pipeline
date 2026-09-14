from pathlib import Path

from ingestion.adapters import CSVAdapter, JSONAdapter, JSONLAdapter, MarkdownAdapter, get_adapter
from ingestion.normalizer import canonicalize_difficulty
from ingestion.models import Difficulty

FIXTURES = Path(__file__).parent / "fixtures"


def test_jsonl_adapter_parses_valid_records_and_skips_bad_ones():
    adapter = JSONLAdapter(source_name="leetcode-sample", license="educational")
    records = list(adapter.parse(FIXTURES / "leetcode_sample.jsonl"))

    # 4 valid lines in fixture; 1 malformed JSON + 1 missing title => skipped
    assert len(records) == 4
    titles = [r.title for r in records]
    assert "Two Sum" in titles
    assert all(r.source_name == "leetcode-sample" for r in records)
    assert all(r.parser == "parse_jsonl_v1" for r in records)


def test_jsonl_adapter_raises_on_missing_file():
    adapter = JSONLAdapter(source_name="x")
    try:
        list(adapter.parse(FIXTURES / "does_not_exist.jsonl"))
        assert False, "expected AdapterError"
    except Exception as exc:
        assert "not found" in str(exc)


def test_csv_adapter_parses_rows():
    adapter = CSVAdapter(source_name="kaggle-sample", license="cc0")
    records = list(adapter.parse(FIXTURES / "kaggle_sample.csv"))

    assert len(records) == 2
    first = records[0]
    assert first.title == "Climbing Stairs"
    assert first.difficulty == "Easy"
    assert "Math" in first.tags


def test_get_adapter_registry():
    assert isinstance(get_adapter("jsonl", source_name="s"), JSONLAdapter)
    assert isinstance(get_adapter("json", source_name="s"), JSONAdapter)


def test_markdown_adapter_parses_doocs_style_problem_dirs():
    adapter = MarkdownAdapter(
        source_name="doocs-leetcode", license="cc-by-sa-4.0", source_url="https://github.com/doocs/leetcode"
    )
    records = {r.title: r for r in adapter.parse(FIXTURES / "doocs_sample")}

    # The category index page (solution/README.md, no problem:start marker)
    # must be skipped, leaving exactly the real problem pages.
    assert len(records) == 4
    assert all(r.parser == "parse_markdown_v1" for r in records.values())
    assert all(r.source_url == "https://github.com/doocs/leetcode" for r in records.values())

    two_sum = records["Two Sum"]
    assert two_sum.difficulty == "Easy"
    assert two_sum.tags == ["Array", "Hash Table"]
    assert two_sum.examples_raw == [
        {
            "input": "nums = [2,7,11,15], target = 9",
            "output": "[0,1]",
            "explanation": "Because nums[0] + nums[1] == 9, we return [0, 1].",
        }
    ]
    assert two_sum.constraints == ["2 <= nums.length <= 10^4", "Only one valid answer exists."]
    assert two_sum.solution_language == "python"
    assert "def twoSum" in two_sum.solution_code

    # Chinese-only README (lcof problem set ships no README_EN.md) still
    # parses, with a raw Chinese difficulty label the normalizer can map.
    zh_problem = records["数组中重复的数字"]
    assert zh_problem.difficulty == "简单"
    assert canonicalize_difficulty(zh_problem.difficulty) == Difficulty.EASY
    assert zh_problem.examples_raw[0]["input"] == "[2, 3, 1, 0, 2, 5, 3]"
    assert "findRepeatNumber" in zh_problem.solution_code


def test_markdown_adapter_flags_premium_problems_and_strips_lock_marker():
    adapter = MarkdownAdapter(source_name="doocs-leetcode", license="cc-by-sa-4.0")
    records = {r.title: r for r in adapter.parse(FIXTURES / "doocs_sample")}

    # The 🔒 suffix must be stripped from the title and surfaced as a
    # structured flag instead of leaking into the canonical title text.
    locked_no_solution = records["Design Excel Sum Formula"]
    assert "🔒" not in locked_no_solution.title
    assert locked_no_solution.is_premium is True
    assert locked_no_solution.solution_code is None

    locked_with_solution = records["Swap Sex of Employees"]
    assert "🔒" not in locked_with_solution.title
    assert locked_with_solution.is_premium is True
    assert locked_with_solution.solution_code is not None

    unlocked = records["Two Sum"]
    assert unlocked.is_premium is False


def test_markdown_adapter_raises_on_missing_directory():
    adapter = MarkdownAdapter(source_name="x")
    try:
        list(adapter.parse(FIXTURES / "does_not_exist_dir"))
        assert False, "expected AdapterError"
    except Exception as exc:
        assert "not found" in str(exc)


def test_get_adapter_registry_includes_markdown():
    assert isinstance(get_adapter("markdown", source_name="s"), MarkdownAdapter)


def test_json_adapter_parses_dict_with_questions(tmp_path):
    json_file = tmp_path / "problems.json"
    import json
    data = {
        "questions": [
            {
                "title": "Two Sum",
                "difficulty": "Easy",
                "topics": ["Array", "Hash Table"],
                "description": "Find two numbers",
                "examples": [{"example_text": "Input: nums = [2,7], target = 9\nOutput: [0,1]"}],
            }
        ]
    }
    json_file.write_text(json.dumps(data))
    adapter = JSONAdapter(source_name="json-test")
    records = list(adapter.parse(json_file))
    assert len(records) == 1
    assert records[0].title == "Two Sum"
    assert records[0].examples_raw[0]["input"] == "nums = [2,7], target = 9"
