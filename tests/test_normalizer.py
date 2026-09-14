from ingestion.models import Difficulty, RawProblem
from ingestion.normalizer import (
    canonicalize_difficulty,
    canonicalize_tags,
    clean_text,
    extract_examples,
    normalize,
)


def test_clean_text_strips_html_and_whitespace():
    dirty = "<p>Hello&nbsp;world.</p>\n\n\n\nExtra   spaces."
    cleaned = clean_text(dirty)
    assert "<p>" not in cleaned
    assert "  " not in cleaned
    assert "Hello" in cleaned


def test_canonicalize_difficulty_variants():
    assert canonicalize_difficulty("Easy") == Difficulty.EASY
    assert canonicalize_difficulty("MEDIUM") == Difficulty.MEDIUM
    assert canonicalize_difficulty("h") == Difficulty.HARD
    assert canonicalize_difficulty(None) == Difficulty.UNKNOWN
    assert canonicalize_difficulty("bogus") == Difficulty.UNKNOWN


def test_canonicalize_tags_dedupes_and_maps_aliases():
    tags = canonicalize_tags(["Array", "HashMap", "arrays", "Hash Table"])
    assert "array" in tags
    assert "hash-map" in tags
    assert len(tags) == len(set(tags))


def test_canonicalize_tags_falls_back_to_unknown():
    assert canonicalize_tags([]) == ["unknown"]


def test_extract_examples_handles_missing_fields():
    raw = [{"input": "1", "output": "2"}, {"input": "3"}, "not a dict"]
    examples = extract_examples(raw)
    assert len(examples) == 1
    assert examples[0].input == "1"
    assert examples[0].output == "2"


def test_normalize_produces_canonical_record():
    raw = RawProblem(
        title="  Two Sum  ",
        description="<p>Some description text that is long enough.</p>",
        difficulty="Easy",
        tags=["Array"],
        examples_raw=[{"input": "1", "output": "2"}],
        constraints=["n <= 10"],
        solution_code="def f(): pass",
        solution_language="python",
        source_name="test-source",
        source_url="http://example.com",
        license="mit",
        raw_file="fixtures/x.jsonl",
        raw_offset=3,
        parser="parse_jsonl_v1",
    )
    record = normalize(raw, ingest_id="ingest-test-1")

    assert record.title == "Two Sum"
    assert record.difficulty == Difficulty.EASY
    assert record.tags == ["array"]
    assert record.has_solution is True
    assert record.canonical_solution is not None
    assert record.canonical_solution.languages == ["python"]
    assert record.canonical_solution.code == "def f(): pass"
    assert record.license_meta is not None
    assert record.license_meta.license == "mit"
    assert record.license_meta.ingest_id == "ingest-test-1"
    assert record.ingest_trace is not None
    assert record.ingest_trace.raw_offset == 3
    assert record.history[-1].step == "normalized"
    assert record.is_premium is False
    assert record.data_unavailable is False


def test_normalize_flags_data_unavailable_only_when_premium_and_no_solution():
    base_kwargs = dict(
        title="Design Excel Sum Formula",
        description="A premium-only problem description that is long enough to pass validation.",
        difficulty="Hard",
        tags=["Design"],
        source_name="doocs-leetcode",
        raw_file="fixtures/premium",
        parser="parse_markdown_v1",
    )

    premium_no_solution = RawProblem(**base_kwargs, is_premium=True, solution_code=None)
    record = normalize(premium_no_solution, ingest_id="ingest-test-2")
    assert record.is_premium is True
    assert record.data_unavailable is True

    premium_with_solution = RawProblem(
        **base_kwargs, is_premium=True, solution_code="SELECT 1", solution_language="sql"
    )
    record = normalize(premium_with_solution, ingest_id="ingest-test-2")
    assert record.is_premium is True
    assert record.data_unavailable is False

    non_premium = RawProblem(**base_kwargs, is_premium=False, solution_code=None)
    record = normalize(non_premium, ingest_id="ingest-test-2")
    assert record.is_premium is False
    assert record.data_unavailable is False
