import json

from transform.llm import MockLLMClient
from transform.pipeline import generate_templates_for_records
from transform.viewer import generate_transform_dashboard, load_run_data

from .fixtures import GOOD_TWO_SUM_REWRITE, NO_SOLUTION_RECORD, TWO_SUM_RECORD


def test_load_run_data_joins_templates_with_canonical_records(tmp_path):
    db_path = tmp_path / "test.db"
    unified_path = tmp_path / "unified_dataset.json"
    unified_path.write_text(json.dumps([TWO_SUM_RECORD, NO_SOLUTION_RECORD]))

    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))
    generate_templates_for_records([TWO_SUM_RECORD], db_path=db_path, llm_client=client, variant_count=3)

    run_data = load_run_data(db_path, unified_path, run_metadata={"metrics": {"accepted": 1}, "rejections": []})

    assert len(run_data["templates"]) == 1
    t = run_data["templates"][0]
    assert t["title"] == "Transaction Pair Match"
    assert t["canonical_title"] == "Two Sum"
    assert t["canonical_difficulty"] == "easy"
    assert len(t["variants"]) == 3
    assert t["canonical_solution_code"]


def test_load_run_data_includes_rejections_with_canonical_context(tmp_path):
    db_path = tmp_path / "test.db"
    unified_path = tmp_path / "unified_dataset.json"
    unified_path.write_text(json.dumps([NO_SOLUTION_RECORD]))

    run_metadata = {
        "metrics": {},
        "rejections": [
            {"canonical_problem_id": NO_SOLUTION_RECORD["id"], "reason": ["no python canonical solution available"]}
        ],
    }
    run_data = load_run_data(db_path, unified_path, run_metadata=run_metadata)

    assert len(run_data["rejections"]) == 1
    rej = run_data["rejections"][0]
    assert rej["canonical_title"] == "Locked Problem"
    assert rej["reasons"] == ["no python canonical solution available"]


def test_generate_transform_dashboard_produces_valid_html(tmp_path):
    db_path = tmp_path / "test.db"
    unified_path = tmp_path / "unified_dataset.json"
    unified_path.write_text(json.dumps([TWO_SUM_RECORD]))

    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))
    generate_templates_for_records([TWO_SUM_RECORD], db_path=db_path, llm_client=client, variant_count=2)

    run_data = load_run_data(db_path, unified_path, run_metadata={"metrics": {"accepted": 1}, "rejections": []})
    out_path = tmp_path / "dashboard.html"
    result_path = generate_transform_dashboard(run_data, output_path=out_path)

    assert result_path == out_path
    content = out_path.read_text(encoding="utf-8")
    assert content.strip().startswith("<!DOCTYPE html>")
    assert content.strip().endswith("</html>")
    assert "Transaction Pair Match" in content
    assert "Two Sum" in content


def test_generate_transform_dashboard_handles_empty_run(tmp_path):
    run_data = {"templates": [], "rejections": [], "run_metadata": {}}
    out_path = tmp_path / "empty_dashboard.html"
    generate_transform_dashboard(run_data, output_path=out_path)

    content = out_path.read_text(encoding="utf-8")
    assert content.strip().endswith("</html>")
    assert "const templates = [];" in content
