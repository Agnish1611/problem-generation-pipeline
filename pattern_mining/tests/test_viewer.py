"""Tests for pattern_mining.viewer."""
import json
from pathlib import Path

import pytest

from pattern_mining.viewer import (
    generate_html_dashboard,
    generate_markdown_report,
    load_patterns_and_problems,
    render_pattern_detail,
    render_terminal_table,
)


@pytest.fixture
def sample_dataset(tmp_path: Path):
    patterns_data = {
        "summary": {
            "n_records": 3,
            "clustered_records": 2,
            "outliers": 1,
            "outlier_rate": 0.333,
            "n_clusters": 1,
        },
        "patterns": [
            {
                "pattern_id": "pattern_0",
                "label": "Binary Search Trees",
                "count": 2,
                "representative": "prob_1",
                "top_tags": ["binary-search", "tree"],
                "difficulty_distribution": {"easy": 1, "medium": 1, "hard": 0},
                "members": ["prob_1", "prob_2"],
            }
        ],
    }

    problems_data = [
        {
            "id": "prob_1",
            "title": "Two Sum BST",
            "difficulty": "easy",
            "description": "Given a BST and target, find two elements.",
            "tags": ["tree", "binary-search"],
            "examples": [{"input": "root = [5,3,6], k = 9", "output": "true", "explanation": "5 + 4 = 9"}],
            "constraints": ["1 <= nodes <= 10^4"],
            "canonical_solution": "def twoSumBST(root, k): pass",
            "source_list": ["leetcode"],
            "template_source_id": "leetcode_653",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "boolean"},
            "schema_inference_confidence": 0.95,
            "fingerprints": {"content_hash": "abcd1234efgh"},
        },
        {
            "id": "prob_2",
            "title": "Search in BST",
            "difficulty": "medium",
            "description": "Search node in BST.",
            "tags": ["tree"],
            "examples": [{"input": "root = [4,2,7], val = 2", "output": "[2,1,3]"}],
            "constraints": ["1 <= nodes <= 5000"],
            "canonical_solution": "def searchBST(root, val): pass",
            "source_list": ["leetcode"],
            "template_source_id": "leetcode_700",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
            "schema_inference_confidence": 0.9,
            "fingerprints": {"content_hash": "efgh5678ijkl"},
        },
        {
            "id": "prob_3",
            "title": "Outlier Problem",
            "difficulty": "hard",
            "description": "A completely isolated puzzle.",
            "tags": ["game-theory"],
            "examples": [],
            "constraints": [],
            "source_list": ["custom"],
            "fingerprints": {"content_hash": "outlier999"},
        },
    ]

    patterns_file = tmp_path / "patterns.json"
    patterns_file.write_text(json.dumps(patterns_data))

    unified_file = tmp_path / "unified.json"
    unified_file.write_text(json.dumps(problems_data))

    return patterns_file, unified_file


def test_load_patterns_and_problems(sample_dataset):
    patterns_file, unified_file = sample_dataset
    summary, patterns, problems_by_id = load_patterns_and_problems(
        patterns_path=patterns_file,
        unified_path=unified_file,
    )

    assert summary["n_records"] == 3
    assert len(patterns) == 1
    assert patterns[0]["pattern_id"] == "pattern_0"
    assert len(problems_by_id) == 3
    assert "prob_1" in problems_by_id


def test_render_terminal_table(sample_dataset):
    patterns_file, unified_file = sample_dataset
    summary, patterns, problems_by_id = load_patterns_and_problems(
        patterns_path=patterns_file,
        unified_path=unified_file,
    )

    table = render_terminal_table(patterns, problems_by_id)
    assert "pattern_0" in table
    assert "Two Sum BST" in table
    assert "Binary Search Trees" in table


def test_render_pattern_detail(sample_dataset):
    patterns_file, unified_file = sample_dataset
    summary, patterns, problems_by_id = load_patterns_and_problems(
        patterns_path=patterns_file,
        unified_path=unified_file,
    )

    detail = render_pattern_detail("pattern_0", patterns, problems_by_id)
    assert "PATTERN DETAILS: pattern_0 — Binary Search Trees" in detail
    assert "Representative : Two Sum BST" in detail
    assert "Member Problems (2 total)" in detail
    assert "Search in BST" in detail


def test_generate_markdown_report(sample_dataset, tmp_path: Path):
    patterns_file, unified_file = sample_dataset
    summary, patterns, problems_by_id = load_patterns_and_problems(
        patterns_path=patterns_file,
        unified_path=unified_file,
    )

    out_md = tmp_path / "report.md"
    generated = generate_markdown_report(summary, patterns, problems_by_id, output_path=out_md)
    assert generated.exists()
    content = generated.read_text()
    assert "# Algorithmic Patterns Report" in content
    assert "Two Sum BST" in content


def test_generate_html_dashboard(sample_dataset, tmp_path: Path):
    patterns_file, unified_file = sample_dataset
    summary, patterns, problems_by_id = load_patterns_and_problems(
        patterns_path=patterns_file,
        unified_path=unified_file,
    )

    out_html = tmp_path / "dashboard.html"
    generated = generate_html_dashboard(summary, patterns, problems_by_id, output_path=out_html)
    assert generated.exists()
    content = generated.read_text()

    # Check key UI tabs and elements
    assert "Algorithmic Pattern & Problem Explorer" in content
    assert "All Questions" in content
    assert "Outliers" in content
    assert "modal-backdrop" in content

    # Check question data is embedded
    assert "Two Sum BST" in content
    assert "Search in BST" in content
    assert "Outlier Problem" in content
    assert "def twoSumBST" in content
    assert "def searchBST" in content
    assert "copySolutionText" in content
    assert "filterByPattern" in content
