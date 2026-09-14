import json

from transform.llm import MockLLMClient
from transform.reauthor import reauthor_template
from transform.storage import (
    link_pattern_to_template,
    migrate_db,
    read_template,
    read_templates_for_canonical_problem,
    read_variants_for_template,
    write_template,
    write_variants,
)

from .fixtures import GOOD_TWO_SUM_REWRITE, TWO_SUM_RECORD


def _accepted_result():
    client = MockLLMClient(default_response=json.dumps(GOOD_TWO_SUM_REWRITE))
    return reauthor_template(TWO_SUM_RECORD, client, variant_count=3)


def test_migrate_db_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    migrate_db(db_path)
    migrate_db(db_path)  # must not raise on second call


def test_write_and_read_template_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    result = _accepted_result()

    write_template(result.template, db_path)
    fetched = read_template(result.template.template_id, db_path)

    assert fetched is not None
    assert fetched["title"] == "Transaction Pair Match"
    assert fetched["canonical_problem_id"] == TWO_SUM_RECORD["id"]
    assert fetched["needs_review"] is False
    assert fetched["provenance"]["transform_model"] == "MockLLMClient"


def test_write_template_upserts_on_conflict(tmp_path):
    db_path = tmp_path / "test.db"
    result = _accepted_result()

    write_template(result.template, db_path)
    result.template.title = "Updated Title"
    write_template(result.template, db_path)

    fetched = read_template(result.template.template_id, db_path)
    assert fetched["title"] == "Updated Title"


def test_read_template_returns_none_for_missing_id(tmp_path):
    db_path = tmp_path / "test.db"
    migrate_db(db_path)
    assert read_template("does-not-exist", db_path) is None


def test_write_and_read_variants_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    result = _accepted_result()

    write_template(result.template, db_path)
    write_variants(result.variants, db_path)

    variants = read_variants_for_template(result.template.template_id, db_path)
    assert len(variants) == len(result.variants) == 3
    assert all(v["template_id"] == result.template.template_id for v in variants)
    assert all(v["seed"] for v in variants)


def test_write_variants_handles_empty_list_without_error(tmp_path):
    db_path = tmp_path / "test.db"
    write_variants([], db_path)  # must not raise


def test_read_templates_for_canonical_problem(tmp_path):
    db_path = tmp_path / "test.db"
    result = _accepted_result()
    write_template(result.template, db_path)

    found = read_templates_for_canonical_problem(TWO_SUM_RECORD["id"], db_path)
    assert len(found) == 1
    assert found[0]["template_id"] == result.template.template_id

    none_found = read_templates_for_canonical_problem("nonexistent-id", db_path)
    assert none_found == []


def test_link_pattern_to_template_noop_when_patterns_table_missing(tmp_path):
    db_path = tmp_path / "test.db"
    # Must not raise even though pattern_mining has never run against
    # this DB (no `patterns` table exists yet).
    link_pattern_to_template("pattern_0", "some-template-id", db_path)


def test_link_pattern_to_template_sets_column(tmp_path):
    import sqlite3

    db_path = tmp_path / "test.db"
    migrate_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE patterns (pattern_id TEXT PRIMARY KEY, label TEXT, size INTEGER)"
        )
        conn.execute("INSERT INTO patterns (pattern_id, label, size) VALUES ('pattern_0', 'test', 5)")
        conn.commit()

    link_pattern_to_template("pattern_0", "template-abc", db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT template_id FROM patterns WHERE pattern_id = 'pattern_0'").fetchone()
    assert row[0] == "template-abc"
