import json
from pathlib import Path

from ingestion.adapters import CSVAdapter, JSONLAdapter
from ingestion.pipeline import IngestSource, IngestionPipeline
from ingestion.storage import Storage

FIXTURES = Path(__file__).parent / "fixtures"


def test_pipeline_end_to_end(tmp_path):
    db_path = tmp_path / "ingestion.db"
    output_path = tmp_path / "unified_dataset.json"

    sources = [
        IngestSource(
            path=str(FIXTURES / "leetcode_sample.jsonl"),
            adapter=JSONLAdapter(source_name="leetcode-sample", license="educational"),
        ),
        IngestSource(
            path=str(FIXTURES / "kaggle_sample.csv"),
            adapter=CSVAdapter(source_name="kaggle-sample", license="cc0"),
        ),
    ]

    pipeline = IngestionPipeline(db_path=db_path)
    metrics = pipeline.run(sources, output_json=output_path, ingest_id="ingest-test-e2e")

    # 4 valid JSONL records + 2 CSV records parsed; 2 of the JSONL are near-dupes of "Two Sum".
    assert metrics.records_parsed == 6
    assert metrics.duplicates_found >= 1
    assert metrics.records_stored == metrics.records_parsed - metrics.duplicates_found
    assert metrics.errors == 0

    # Unified dataset export exists and is valid JSON with the expected shape.
    assert output_path.exists()
    data = json.loads(output_path.read_text())
    assert len(data) == metrics.records_stored
    titles = {d["title"] for d in data}
    assert "Two Sum" in titles
    assert "Climbing Stairs" in titles

    for d in data:
        assert d["license_meta"] is not None
        assert d["ingest_trace"] is not None
        assert d["fingerprints"]["content_hash"]

    # Solution code from the source (JSONL "solution" field) must survive
    # the full pipeline end-to-end into the exported unified dataset.
    two_sum = next(d for d in data if d["title"] == "Two Sum")
    assert two_sum["canonical_solution"] is not None
    assert "def two_sum" in two_sum["canonical_solution"]["code"]

    # Storage persisted both surviving and deduplicated records for audit.
    with Storage(db_path) as storage:
        all_including_dupes = storage.all_records(include_deduplicated=True)
        assert len(all_including_dupes) == metrics.records_parsed
        surviving_only = storage.all_records(include_deduplicated=False)
        assert len(surviving_only) == metrics.records_stored


def test_pipeline_is_idempotent_on_rerun(tmp_path):
    db_path = tmp_path / "ingestion.db"
    output_path = tmp_path / "unified_dataset.json"

    def make_sources():
        return [
            IngestSource(
                path=str(FIXTURES / "leetcode_sample.jsonl"),
                adapter=JSONLAdapter(source_name="leetcode-sample", license="educational"),
            )
        ]

    pipeline = IngestionPipeline(db_path=db_path)
    metrics1 = pipeline.run(make_sources(), output_json=output_path, ingest_id="run-1")
    metrics2 = pipeline.run(make_sources(), output_json=output_path, ingest_id="run-2")

    with Storage(db_path) as storage:
        all_records = storage.all_records(include_deduplicated=True)
        # Re-running the same source file should not create new rows.
        assert len(all_records) == metrics1.records_parsed

    assert metrics2.records_parsed == metrics1.records_parsed
