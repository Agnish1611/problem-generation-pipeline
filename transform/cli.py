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
from .viewer import generate_transform_dashboard, load_run_data

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transform", description="Template Re-authoring Pipeline CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging to stderr")
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Also write logs to this file (always at DEBUG level, rotates at 50 MB)",
    )

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
    run_parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="LLM call timeout in seconds per attempt (default: 300 — increase for slow/large models)"
    )
    run_parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max tokens the LLM may generate per call (default: 512)"
    )
    think_group = run_parser.add_mutually_exclusive_group()
    think_group.add_argument(
        "--no-think", dest="think", action="store_false", default=False,
        help="(default) Disable chain-of-thought for thinking models like qwen3/deepseek-r1"
    )
    think_group.add_argument(
        "--think", dest="think", action="store_true",
        help="Enable chain-of-thought reasoning (slower; only useful if you want to inspect the model's reasoning)"
    )

    view_parser = subparsers.add_parser("view", help="View accepted templates from the DB")
    view_parser.add_argument("--db", default="data/output/ingestion.db", help="Path to SQLite database")
    view_parser.add_argument(
        "--input", default="data/output/unified_dataset.json", help="Path to unified dataset JSON (for canonical context)"
    )
    view_parser.add_argument(
        "--export-html", metavar="PATH", default=None,
        help="Generate an HTML dashboard (default: data/output/transform_dashboard.html)",
    )
    view_parser.add_argument("--list", action="store_true", help="Print a summary table of all accepted templates to stdout")

    return parser


def _configure_logging(verbose: bool, log_file: str | None) -> None:
    """Set up root logger with optional file handler."""
    import logging.handlers

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter by their own level

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    # stderr handler
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stderr_handler.setFormatter(fmt)
    root.addHandler(stderr_handler)

    # optional file handler (rotating, always DEBUG)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logging.getLogger(__name__).info("Logging to file: %s", log_file)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_logging(args.verbose, args.log_file)

    if args.command == "run":
        client = get_default_client(think=args.think)
        logger.info(
            "Using LLM client: %s (think=%s)",
            client.__class__.__name__,
            getattr(client, 'think', 'n/a'),
        )

        metrics = generate_templates_for_patterns(
            unified_json_path=args.input,
            db_path=args.db,
            llm_client=client,
            limit=args.limit,
            force=args.force,
            max_attempts=args.max_attempts,
            variant_count=args.variant_count,
            llm_timeout=args.timeout,
            llm_max_tokens=args.max_tokens,
        )

        print(json.dumps(metrics.as_dict(), indent=2))
        if metrics.rejections:
            print(f"\n{len(metrics.rejections)} rejection(s) — first 10:", file=sys.stderr)
            for rejection in metrics.rejections[:10]:
                print(f"  {rejection}", file=sys.stderr)
        return 0

    if args.command == "view":
        run_data = load_run_data(db_path=args.db, unified_json_path=args.input)
        templates = run_data["templates"]

        if args.list or (not args.export_html):
            if not templates:
                print("No accepted templates in the database yet.")
            else:
                print(f"{'#':<4} {'Title':<45} {'Difficulty':<10} {'Variants':<9} {'Review?':<8} {'Created'}")
                print("-" * 100)
                for i, t in enumerate(templates, 1):
                    review = "⚠ yes" if t["needs_review"] else "ok"
                    print(
                        f"{i:<4} {t['title'][:44]:<45} "
                        f"{t['canonical_difficulty']:<10} {len(t['variants']):<9} "
                        f"{review:<8} {t['created_at'][:19]}"
                    )
                print(f"\nTotal: {len(templates)} template(s)")

        if args.export_html is not None:
            out_path = args.export_html or "data/output/transform_dashboard.html"
            path = generate_transform_dashboard(run_data, output_path=out_path)
            print(f"Dashboard written to: {path}")
            logger.info("Dashboard written to: %s", path)

        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
