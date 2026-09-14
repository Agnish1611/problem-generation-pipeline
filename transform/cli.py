"""CLI entrypoint for the Template Re-authoring Pipeline.

Usage:
    python -m transform.cli run \\
        --input data/output/unified_dataset.json \\
        --db data/output/ingestion.db \\
        --limit 10

By default uses the mock LLM client (TRANSFORM_LLM_MODE unset/"mock") so
this is safe to run without any local model installed — it will just
reject every record with a generic "mock client has no configured
response" JSON parse failure. Set TRANSFORM_LLM_MODE=ollama (and
TRANSFORM_LLM_MODEL) to use a real local model; see transform/llm.py's
module docstring for setup instructions.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .llm import get_default_client
from .pipeline import generate_templates_for_patterns
from .reauthor import DEFAULT_MAX_ATTEMPTS
from .validator import DEFAULT_VARIANT_COUNT

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transform", description="Template Re-authoring Pipeline CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Generate templates for canonical problems")
    run_parser.add_argument(
        "--input", default="data/output/unified_dataset.json", help="Path to unified dataset JSON"
    )
    run_parser.add_argument("--db", default="data/output/ingestion.db", help="Path to SQLite database")
    run_parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N records (for smoke-testing)"
    )
    run_parser.add_argument(
        "--force", action="store_true", help="Re-run and overwrite records that already have a template"
    )
    run_parser.add_argument(
        "--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help="Max LLM re-prompt attempts per record"
    )
    run_parser.add_argument(
        "--variant-count", type=int, default=DEFAULT_VARIANT_COUNT, help="Randomized variants to generate per accepted template"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "run":
        client = get_default_client()
        logger.info("Using LLM client: %s", client.__class__.__name__)

        metrics = generate_templates_for_patterns(
            unified_json_path=args.input,
            db_path=args.db,
            llm_client=client,
            limit=args.limit,
            force=args.force,
            max_attempts=args.max_attempts,
            variant_count=args.variant_count,
        )

        print(json.dumps(metrics.as_dict(), indent=2))
        if metrics.rejections:
            print(f"\n{len(metrics.rejections)} rejection(s) — first 10:", file=sys.stderr)
            for rejection in metrics.rejections[:10]:
                print(f"  {rejection}", file=sys.stderr)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
