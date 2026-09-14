import json

from ingestion.models import CanonicalRecord, IngestTrace
from ingestion.storage import Storage, export_unified_dataset


def _record(title, raw_file="f.jsonl", offset=0, source="src"):
    return CanonicalRecord(
        title=title,
        description="desc",
        source_list=[source],
        ingest_trace=IngestTrace(parser="p", raw_file=raw_file, raw_offset=offset),
    )


def test_upsert_and_retrieve_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    with Storage(db_path) as storage:
        record = _record("Two Sum")
        storage.upsert(record)
        storage.conn.commit()

        all_records = storage.all_records()
        assert len(all_records) == 1
        assert all_records[0].title == "Two Sum"


def test_upsert_is_idempotent_on_same_source_file_offset(tmp_path):
    db_path = tmp_path / "test.db"
    with Storage(db_path) as storage:
        record1 = _record("Two Sum", raw_file="f.jsonl", offset=5)
        storage.upsert(record1)
        storage.conn.commit()

        record2 = _record("Two Sum (updated title)", raw_file="f.jsonl", offset=5)
        storage.upsert(record2)
        storage.conn.commit()

        all_records = storage.all_records()
        assert len(all_records) == 1
        assert all_records[0].title == "Two Sum (updated title)"
        assert all_records[0].version >= 2


def test_export_unified_dataset_writes_valid_json(tmp_path):
    out_path = tmp_path / "unified_dataset.json"
    records = [_record("A"), _record("B", offset=1)]
    export_unified_dataset(records, out_path)

    data = json.loads(out_path.read_text())
    assert len(data) == 2
    assert {d["title"] for d in data} == {"A", "B"}


def test_solution_code_survives_sqlite_and_json_export_roundtrip(tmp_path):
    from ingestion.models import CanonicalSolution

    db_path = tmp_path / "test.db"
    out_path = tmp_path / "unified_dataset.json"

    record = _record("Two Sum")
    record.has_solution = True
    record.canonical_solution = CanonicalSolution(languages=["python"], code="def two_sum(): pass")

    with Storage(db_path) as storage:
        storage.upsert(record)
        storage.conn.commit()

        stored = storage.all_records()[0]
        assert stored.canonical_solution is not None
        assert stored.canonical_solution.code == "def two_sum(): pass"

    export_unified_dataset([record], out_path)
    data = json.loads(out_path.read_text())
    assert data[0]["canonical_solution"]["code"] == "def two_sum(): pass"
