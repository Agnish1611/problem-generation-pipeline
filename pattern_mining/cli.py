"""CLI for the pattern mining module.

Supports two commands:
  1. run: executes the full end-to-end pattern mining pipeline on unified_dataset.json.
  2. assign: incrementally assigns new problem records (from JSONL or JSON) to patterns.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

from .assigner import bulk_assign_new_records
from .pipeline import run_full_pipeline

logger = logging.getLogger("pattern_mining")


def _load_problems_file(file_path: Path) -> List[dict[str, Any]]:
    """Load problem records from either JSON or JSONL file format."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    records: List[dict[str, Any]] = []
    suffix = file_path.suffix.lower()

    if suffix in [".jsonl", ".ndjson"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as err:
                        logger.warning("Line %d in %s is invalid JSON: %s", line_idx, file_path, err)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
            else:
                records = [data]

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pattern_mining",
        description="Algorithmic Pattern Mining and Incremental Assignment CLI",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run full pattern mining pipeline")
    run_parser.add_argument(
        "--input",
        default="data/output/unified_dataset.json",
        help="Path to unified dataset JSON (default: data/output/unified_dataset.json)",
    )
    run_parser.add_argument(
        "--db",
        default="data/output/ingestion.db",
        help="Path to SQLite database (default: data/output/ingestion.db)",
    )
    run_parser.add_argument(
        "--output",
        default="data/output/patterns.json",
        help="Path to output patterns JSON (default: data/output/patterns.json)",
    )
    run_parser.add_argument(
        "--cache-dir",
        default="data/processed",
        help="Directory for caching intermediate embeddings (default: data/processed)",
    )
    run_parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Force recomputation of embeddings even if cached",
    )
    run_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=10,
        help="HDBSCAN min_cluster_size parameter (default: 10)",
    )
    run_parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="HDBSCAN min_samples parameter (default: 1)",
    )
    run_parser.add_argument(
        "--umap-dim",
        type=int,
        default=64,
        help="Target dimension for UMAP reduction (default: 64)",
    )

    # Assign subcommand
    assign_parser = subparsers.add_parser("assign", help="Incrementally assign new problems to patterns")
    assign_parser.add_argument(
        "--input",
        required=True,
        help="Path to new records file (JSON or JSONL)",
    )
    assign_parser.add_argument(
        "--db",
        default="data/output/ingestion.db",
        help="Path to SQLite database (default: data/output/ingestion.db)",
    )
    assign_parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Cosine similarity threshold for pattern assignment (default: 0.80)",
    )
    assign_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation (default: 64)",
    )

    # View subcommand
    view_parser = subparsers.add_parser("view", help="View discovered patterns in readable format")
    view_parser.add_argument(
        "--patterns",
        default="data/output/patterns.json",
        help="Path to patterns JSON file (default: data/output/patterns.json)",
    )
    view_parser.add_argument(
        "--unified",
        default="data/output/unified_dataset.json",
        help="Path to unified dataset JSON (default: data/output/unified_dataset.json)",
    )
    view_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of patterns to display in terminal table (default: 25, 0 for all)",
    )
    view_parser.add_argument(
        "--pattern",
        default=None,
        help="Inspect a specific pattern ID in detail with all its member problems (e.g. pattern_0)",
    )
    view_parser.add_argument(
        "--search",
        default=None,
        help="Search patterns by label, tag, or problem name keyword",
    )
    view_parser.add_argument(
        "--export-html",
        nargs="?",
        const="data/output/patterns_dashboard.html",
        default=None,
        help="Generate an interactive HTML dashboard (default: data/output/patterns_dashboard.html)",
    )
    view_parser.add_argument(
        "--export-markdown",
        nargs="?",
        const="data/output/patterns_report.md",
        default=None,
        help="Generate a Markdown report (default: data/output/patterns_report.md)",
    )

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    )

    if args.command == "run":
        cluster_params = {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "umap_dim": args.umap_dim,
        }
        try:
            metrics = run_full_pipeline(
                unified_json_path=args.input,
                db_path=args.db,
                output_patterns_path=args.output,
                cache_dir=args.cache_dir,
                recompute_embeddings=args.recompute_embeddings,
                cluster_params=cluster_params,
            )
            print(json.dumps(metrics, indent=2))
            return 0
        except Exception as exc:
            logger.exception("Pattern mining pipeline failed: %s", exc)
            return 1

    elif args.command == "assign":
        input_path = Path(args.input)
        try:
            problems = _load_problems_file(input_path)
            logger.info("Loaded %d problems from %s", len(problems), input_path)

            assignments = bulk_assign_new_records(
                list_of_problems=problems,
                db_path=args.db,
                batch_size=args.batch_size,
                threshold=args.threshold,
            )

            result_summary = {
                "input_records": len(problems),
                "assigned_records": len(assignments),
                "threshold": args.threshold,
                "db_path": str(args.db),
            }
            print(json.dumps(result_summary, indent=2))
            return 0
        except Exception as exc:
            logger.exception("Incremental assignment failed: %s", exc)
            return 1

    elif args.command == "view":
        try:
            from .viewer import (
                generate_html_dashboard,
                generate_markdown_report,
                load_patterns_and_problems,
                render_pattern_detail,
                render_terminal_table,
            )

            summary, patterns, problems_by_id = load_patterns_and_problems(
                patterns_path=args.patterns,
                unified_path=args.unified,
            )

            # Export HTML dashboard if requested
            if args.export_html:
                out_html = generate_html_dashboard(summary, patterns, problems_by_id, output_path=args.export_html)
                print(f"Generated interactive HTML dashboard at: {out_html}")

            # Export Markdown report if requested
            if args.export_markdown:
                out_md = generate_markdown_report(summary, patterns, problems_by_id, output_path=args.export_markdown)
                print(f"Generated Markdown report at: {out_md}")

            # Detail view for single pattern
            if args.pattern:
                print(render_pattern_detail(args.pattern, patterns, problems_by_id))
            elif not args.export_html and not args.export_markdown:
                # Default: render terminal table
                print(render_terminal_table(
                    patterns=patterns,
                    problems_by_id=problems_by_id,
                    limit=args.limit,
                    search_query=args.search,
                ))

            return 0
        except Exception as exc:
            logger.exception("Failed to view patterns: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
