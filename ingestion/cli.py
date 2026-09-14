"""CLI entrypoint for the Dataset Ingestion Layer.

Usage:
    ingest --source jsonl:path/to/file.jsonl:source-name:license \\
           --source csv:path/to/file.csv:other-source:mit \\
           --source markdown:path/to/doocs_clone:doocs-leetcode:cc-by-sa-4.0 \\
           --db data/output/ingestion.db \\
           --output data/output/unified_dataset.json

Each --source is "<kind>:<path>:<source_name>[:<license>[:<source_url>]]"
where kind is one of: json, jsonl, csv, html, markdown.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .adapters import get_adapter
from .pipeline import IngestSource, IngestionPipeline


def _parse_source_spec(spec: str) -> IngestSource:
    # maxsplit=4 caps at 5 fields (kind:path:source_name:license:source_url)
    # so a source_url containing its own colons (e.g. "https://...") is
    # captured whole in the last field instead of being chopped up.
    parts = spec.split(":", 4)
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            f"Invalid --source spec {spec!r}. Expected kind:path:source_name[:license[:source_url]]"
        )
    kind, path, source_name, *rest = parts
    license_name = rest[0] if len(rest) >= 1 and rest[0] else "unknown"
    source_url = rest[1] if len(rest) >= 2 and rest[1] else None
    adapter = get_adapter(kind, source_name=source_name, license=license_name, source_url=source_url)
    return IngestSource(path=path, adapter=adapter)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description="Dataset Ingestion Layer CLI")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        dest="sources",
        help="Source spec: kind:path:source_name[:license[:source_url]]. Repeatable.",
    )
    parser.add_argument("--db", default="data/output/ingestion.db", help="SQLite DB path")
    parser.add_argument(
        "--output", default="data/output/unified_dataset.json", help="Path for unified_dataset.json export"
    )
    parser.add_argument("--ingest-id", default=None, help="Override the auto-generated ingest id")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        sources = [_parse_source_spec(spec) for spec in args.sources]
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pipeline = IngestionPipeline(db_path=args.db)
    metrics = pipeline.run(sources, output_json=args.output, ingest_id=args.ingest_id)

    print(json.dumps(metrics.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
