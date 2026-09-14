"""Human-readable viewer and report generator for pattern mining outputs.

Provides rich terminal table rendering, pattern search, pattern inspection,
and interactive HTML dashboard export for exploring discovered algorithmic patterns.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_patterns_and_problems(
    patterns_path: str | Path = "data/output/patterns.json",
    unified_path: str | Path = "data/output/unified_dataset.json",
) -> tuple[dict[str, Any], List[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load patterns.json and index unified_dataset.json by problem_id."""
    p_path = Path(patterns_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {p_path}. Run the pipeline first.")

    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    patterns = data.get("patterns", [])

    problems_by_id: dict[str, dict[str, Any]] = {}
    u_path = Path(unified_path)
    if u_path.exists():
        with open(u_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                problems_by_id = {str(p["id"]): p for p in raw if p.get("id")}

    return summary, patterns, problems_by_id


def render_terminal_table(
    patterns: List[dict[str, Any]],
    problems_by_id: dict[str, dict[str, Any]],
    limit: int = 25,
    search_query: Optional[str] = None,
    sort_by: str = "size",
) -> str:
    """Render a clean, formatted ASCII table of patterns for the terminal."""
    filtered = patterns
    if search_query:
        q = search_query.lower()
        filtered = [
            p for p in patterns
            if q in p.get("label", "").lower()
            or any(q in t.lower() for t in p.get("top_tags", []))
            or q in problems_by_id.get(p.get("representative", ""), {}).get("title", "").lower()
            or q in p.get("pattern_id", "").lower()
        ]

    if sort_by == "size":
        filtered = sorted(filtered, key=lambda p: int(p.get("count", p.get("size", 0))), reverse=True)
    elif sort_by == "id":
        filtered = sorted(filtered, key=lambda p: p.get("pattern_id", ""))

    display_items = filtered[:limit] if limit > 0 else filtered

    lines = []
    lines.append(f"\nDiscovered Patterns ({len(filtered)} matching, showing top {len(display_items)}):")
    lines.append("=" * 115)
    lines.append(f"{'Pattern ID':<13} {'Size':<6} {'Representative Problem':<35} {'Difficulty':<16} {'Auto-Generated Label'}")
    lines.append("-" * 115)

    for p in display_items:
        pid = p.get("pattern_id", "")
        size = str(p.get("count", p.get("size", 0)))
        rep_id = p.get("representative", "")
        rep_prob = problems_by_id.get(rep_id, {})
        rep_title = rep_prob.get("title", rep_id)[:33]
        if len(rep_prob.get("title", "")) > 33:
            rep_title += ".."

        diff = p.get("difficulty_distribution", {})
        diff_str = f"E:{diff.get('easy', 0)} M:{diff.get('medium', 0)} H:{diff.get('hard', 0)}"
        label = p.get("label", "")[:40]

        lines.append(f"{pid:<13} {size:<6} {rep_title:<35} {diff_str:<16} {label}")

    lines.append("=" * 115)
    lines.append(f"Tip: Use --pattern <id> to inspect all member problems in a pattern.\n")
    return "\n".join(lines)


def render_pattern_detail(
    pattern_id: str,
    patterns: List[dict[str, Any]],
    problems_by_id: dict[str, dict[str, Any]],
) -> str:
    """Render comprehensive details for a single pattern."""
    target = next((p for p in patterns if p.get("pattern_id") == pattern_id), None)
    if not target:
        return f"Error: Pattern '{pattern_id}' not found."

    rep_id = target.get("representative", "")
    rep_prob = problems_by_id.get(rep_id, {})

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append(f"PATTERN DETAILS: {target.get('pattern_id')} — {target.get('label')}")
    lines.append("=" * 80)
    lines.append(f"  Cluster Size   : {target.get('count', target.get('size', 0))} problems")
    lines.append(f"  Top Tags       : {', '.join(target.get('top_tags', []))}")
    lines.append(f"  Difficulty Dist: {target.get('difficulty_distribution', {})}")
    lines.append(f"  Representative : {rep_prob.get('title', 'Unknown')} (ID: {rep_id})")

    desc = rep_prob.get("description", "").strip()
    if desc:
        desc_snippet = desc[:300].replace("\n", " ")
        lines.append(f"  Problem Summary: {desc_snippet}...")

    members = target.get("members", [])
    lines.append(f"\n  Member Problems ({len(members)} total):")
    lines.append("  " + "-" * 76)
    for idx, mid in enumerate(members, 1):
        prob = problems_by_id.get(mid, {})
        title = prob.get("title", "Unknown Title")
        diff = prob.get("difficulty", "unknown")
        lines.append(f"  {idx:>3}. [{diff:<6}] {title}")

    lines.append("=" * 80 + "\n")
    return "\n".join(lines)


def generate_markdown_report(
    summary: dict[str, Any],
    patterns: List[dict[str, Any]],
    problems_by_id: dict[str, dict[str, Any]],
    output_path: str | Path = "data/output/patterns_report.md",
) -> Path:
    """Generate a clean Markdown summary report of patterns."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sorted_patterns = sorted(patterns, key=lambda p: int(p.get("count", 0)), reverse=True)

    lines = [
        "# Algorithmic Patterns Report",
        "",
        "## Summary Metrics",
        "",
        f"- **Total Problems**: {summary.get('n_records', 'N/A')}",
        f"- **Discovered Patterns**: {summary.get('n_clusters', 'N/A')}",
        f"- **Clustered Problems**: {summary.get('clustered_records', 'N/A')}",
        f"- **Outliers**: {summary.get('outliers', 'N/A')} ({summary.get('outlier_rate', 0):.2%})",
        f"- **Generated At**: {summary.get('generated_at', 'N/A')}",
        "",
        "## Discovered Algorithmic Patterns",
        "",
        "| ID | Size | Representative Problem | Top Tags | Difficulty (E/M/H) | Auto Label |",
        "|---|---|---|---|---|---|",
    ]

    for p in sorted_patterns:
        pid = p.get("pattern_id", "")
        size = p.get("count", 0)
        rep_id = p.get("representative", "")
        rep_title = problems_by_id.get(rep_id, {}).get("title", rep_id).replace("|", "\\|")
        tags = ", ".join(p.get("top_tags", [])[:3])
        diff = p.get("difficulty_distribution", {})
        diff_str = f"{diff.get('easy', 0)} / {diff.get('medium', 0)} / {diff.get('hard', 0)}"
        label = p.get("label", "").replace("|", "\\|")
        lines.append(f"| `{pid}` | {size} | **{rep_title}** | {tags} | {diff_str} | {label} |")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def generate_html_dashboard(
    summary: dict[str, Any],
    patterns: List[dict[str, Any]],
    problems_by_id: dict[str, dict[str, Any]],
    output_path: str | Path = "data/output/patterns_dashboard.html",
) -> Path:
    """Generate a modern, responsive, interactive single-file HTML dashboard

    that includes full question data for all 3,534 problems, pattern exploration,
    search, filtering, and complete question inspection modals.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Map problem_id -> assigned pattern_id
    problem_to_pattern: dict[str, str] = {}
    for p in patterns:
        pat_id = str(p.get("pattern_id", ""))
        for mid in p.get("members", []):
            problem_to_pattern[str(mid)] = pat_id

    # 2. Build full question payload for all problems (excluding massive raw embeddings)
    all_questions: dict[str, dict[str, Any]] = {}
    for pid, prob in problems_by_id.items():
        fp = prob.get("fingerprints") or {}
        content_hash = fp.get("content_hash", "") if isinstance(fp, dict) else ""
        all_questions[pid] = {
            "id": pid,
            "title": prob.get("title", "Untitled"),
            "difficulty": prob.get("difficulty", "unknown"),
            "tags": prob.get("tags", []) or [],
            "description": prob.get("description", "") or "",
            "examples": prob.get("examples", []) or [],
            "constraints": prob.get("constraints", []) or [],
            "input_schema": prob.get("input_schema") or {},
            "output_schema": prob.get("output_schema") or {},
            "canonical_solution": prob.get("canonical_solution", "") or "",
            "sources": prob.get("source_list", []) or [],
            "template_source_id": prob.get("template_source_id", "") or "",
            "confidence": prob.get("schema_inference_confidence"),
            "content_hash": content_hash,
            "pattern_id": problem_to_pattern.get(pid, "Outlier"),
        }

    # 3. Build enriched patterns payload (referencing problem IDs)
    enriched_patterns = []
    for p in sorted(patterns, key=lambda x: int(x.get("count", x.get("size", 0))), reverse=True):
        rep_id = str(p.get("representative", ""))
        enriched_patterns.append({
            "pattern_id": p.get("pattern_id"),
            "label": p.get("label"),
            "size": p.get("count", p.get("size", 0)),
            "top_tags": p.get("top_tags", []),
            "difficulty": p.get("difficulty_distribution", {}),
            "representative_id": rep_id,
            "member_ids": [str(m) for m in p.get("members", [])],
        })

    patterns_json_str = json.dumps(enriched_patterns)
    questions_json_str = json.dumps(all_questions)
    summary_json_str = json.dumps(summary)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Algorithmic Pattern & Problem Explorer</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --surface: #111827;
      --card: #1f2937;
      --card-hover: #283548;
      --card-border: #374151;
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --text-dim: #6b7280;
      --primary: #38bdf8;
      --primary-dim: rgba(56, 189, 248, 0.15);
      --primary-hover: #0284c7;
      --easy: #10b981;
      --easy-bg: rgba(16, 185, 129, 0.15);
      --medium: #f59e0b;
      --medium-bg: rgba(245, 158, 11, 0.15);
      --hard: #ef4444;
      --hard-bg: rgba(239, 68, 68, 0.15);
      --code-bg: #030712;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 0;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--card-border);
      padding: 18px 32px;
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .brand h1 {{ font-size: 1.35rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }}
    .brand .badge {{ font-size: 0.75rem; background: var(--primary-dim); color: var(--primary); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(56,189,248,0.3); }}
    .nav-tabs {{ display: flex; gap: 8px; }}
    .nav-tab {{
      background: transparent;
      border: 1px solid transparent;
      color: var(--text-muted);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }}
    .nav-tab:hover {{ color: var(--text); background: rgba(255, 255, 255, 0.05); }}
    .nav-tab.active {{ color: #000; background: var(--primary); font-weight: 600; border-color: var(--primary); }}

    .main-container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}

    /* Metrics Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px 20px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    .metric-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); font-weight: 600; }}
    .metric-value {{ font-size: 1.85rem; font-weight: 700; color: #fff; margin-top: 4px; }}
    .metric-sub {{ font-size: 0.8rem; color: var(--primary); margin-top: 2px; }}

    /* Controls */
    .toolbar {{
      background: var(--surface);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 20px;
    }}
    .search-box {{ flex: 1; min-width: 280px; position: relative; }}
    .search-input {{
      width: 100%;
      background: var(--card);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.95rem;
    }}
    .search-input:focus {{ outline: none; border-color: var(--primary); }}
    .filter-select {{
      background: var(--card);
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.9rem;
      cursor: pointer;
    }}
    .filter-select:focus {{ outline: none; border-color: var(--primary); }}

    /* Patterns Table */
    .patterns-table {{ width: 100%; border-collapse: separate; border-spacing: 0 8px; }}
    .patterns-table th {{
      text-align: left;
      padding: 8px 16px;
      color: var(--text-muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .pattern-row {{
      background: var(--surface);
      border: 1px solid var(--card-border);
      cursor: pointer;
      transition: all 0.15s;
    }}
    .pattern-row:hover {{ border-color: var(--primary); transform: translateY(-1px); }}
    .pattern-row td {{ padding: 14px 16px; vertical-align: middle; }}
    .pattern-row td:first-child {{ border-top-left-radius: 10px; border-bottom-left-radius: 10px; border-left: 3px solid var(--primary); }}
    .pattern-row td:last-child {{ border-top-right-radius: 10px; border-bottom-right-radius: 10px; }}

    .pattern-id {{ font-weight: 700; color: var(--primary); font-family: ui-monospace, SFMono-Regular, monospace; font-size: 0.95rem; }}
    .pattern-label {{ font-size: 1.05rem; font-weight: 600; color: #fff; }}
    .rep-name {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 3px; display: flex; align-items: center; gap: 6px; }}
    .size-badge {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: #030712;
      border: 1px solid var(--card-border);
      padding: 4px 12px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 0.9rem;
      color: #fff;
    }}
    .tag-badge {{
      display: inline-block;
      background: var(--card);
      color: #d1d5db;
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 6px;
      margin: 2px 4px 2px 0;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .diff-bar {{
      display: flex;
      height: 8px;
      width: 110px;
      border-radius: 4px;
      overflow: hidden;
      background: #374151;
      margin-bottom: 4px;
    }}
    .diff-easy {{ background: var(--easy); }}
    .diff-med {{ background: var(--medium); }}
    .diff-hard {{ background: var(--hard); }}
    .diff-text {{ font-size: 0.75rem; color: var(--text-muted); }}

    /* Questions Grid */
    .questions-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
    }}
    .question-card {{
      background: var(--surface);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 18px;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: all 0.15s;
    }}
    .question-card:hover {{ border-color: var(--primary); transform: translateY(-2px); }}
    .q-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 10px; }}
    .q-title {{ font-size: 1.05rem; font-weight: 600; color: #fff; line-height: 1.35; }}
    .badge-diff {{
      font-size: 0.72rem;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      white-space: nowrap;
    }}
    .badge-diff.easy {{ background: var(--easy-bg); color: var(--easy); border: 1px solid rgba(16,185,129,0.3); }}
    .badge-diff.medium {{ background: var(--medium-bg); color: var(--medium); border: 1px solid rgba(245,158,11,0.3); }}
    .badge-diff.hard {{ background: var(--hard-bg); color: var(--hard); border: 1px solid rgba(239,68,68,0.3); }}
    .badge-diff.unknown {{ background: rgba(156,163,175,0.15); color: #9ca3af; border: 1px solid rgba(156,163,175,0.3); }}
    .badge-pat {{
      font-size: 0.75rem;
      padding: 2px 7px;
      border-radius: 4px;
      background: var(--primary-dim);
      color: var(--primary);
      font-family: monospace;
      border: 1px solid rgba(56,189,248,0.3);
      display: inline-block;
      margin-top: 4px;
    }}
    .badge-pat.outlier {{ background: rgba(156,163,175,0.1); color: var(--text-muted); border-color: var(--card-border); }}
    .q-desc-snippet {{ color: var(--text-muted); font-size: 0.85rem; line-height: 1.4; margin: 10px 0; max-height: 4.2em; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }}
    .q-footer {{ margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255, 255, 255, 0.05); display: flex; justify-content: space-between; align-items: center; }}
    .view-btn {{
      background: var(--card);
      border: 1px solid var(--card-border);
      color: var(--primary);
      font-size: 0.8rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
    }}

    /* Pattern Drawer */
    .drawer {{
      position: fixed;
      top: 0; right: -650px;
      width: 650px; max-width: 95vw;
      height: 100vh;
      background: #0f172a;
      border-left: 1px solid var(--card-border);
      box-shadow: -12px 0 40px rgba(0, 0, 0, 0.7);
      transition: right 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 100;
      display: flex;
      flex-direction: column;
    }}
    .drawer.open {{ right: 0; }}
    .drawer-header {{
      padding: 20px 24px;
      border-bottom: 1px solid var(--card-border);
      background: #1e293b;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .drawer-close {{ background: none; border: none; color: var(--text-muted); font-size: 1.8rem; cursor: pointer; }}
    .drawer-close:hover {{ color: #fff; }}
    .drawer-body {{ padding: 24px; overflow-y: auto; flex: 1; }}

    /* Question Modal (Full Question View) */
    .modal-backdrop {{
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px);
      z-index: 200;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .modal-backdrop.open {{ display: flex; }}
    .modal-dialog {{
      background: #111827;
      border: 1px solid var(--card-border);
      border-radius: 16px;
      width: 900px;
      max-width: 95vw;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
      animation: modalSlide 0.2s ease-out;
    }}
    @keyframes modalSlide {{ from {{ opacity: 0; transform: translateY(15px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .modal-header {{
      padding: 20px 28px;
      border-bottom: 1px solid var(--card-border);
      background: #1f2937;
      border-top-left-radius: 16px;
      border-top-right-radius: 16px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }}
    .modal-body {{
      padding: 28px;
      overflow-y: auto;
      flex: 1;
    }}
    .modal-section-title {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--primary);
      font-weight: 700;
      margin: 20px 0 8px 0;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 4px;
    }}
    .modal-section-title:first-child {{ margin-top: 0; }}
    .desc-text {{
      font-size: 0.96rem;
      color: #e5e7eb;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .example-card {{
      background: var(--code-bg);
      border: 1px solid #1f2937;
      border-radius: 8px;
      padding: 14px 16px;
      margin-bottom: 12px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 0.88rem;
    }}
    .example-title {{ font-weight: 700; color: var(--primary); margin-bottom: 6px; font-size: 0.8rem; text-transform: uppercase; }}
    .example-field {{ margin-bottom: 4px; }}
    .example-field span.lbl {{ color: var(--text-muted); }}
    .example-field span.val {{ color: #a5f3fc; }}
    .constraints-list {{ padding-left: 20px; color: #d1d5db; font-size: 0.92rem; line-height: 1.6; }}
    .raw-json {{
      background: var(--code-bg);
      border: 1px solid #374151;
      padding: 12px;
      border-radius: 8px;
      font-family: monospace;
      font-size: 0.8rem;
      max-height: 250px;
      overflow: auto;
      color: #94a3b8;
    }}

    .code-container {{
      position: relative;
      background: var(--code-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
      margin-top: 8px;
    }}
    .code-toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 14px;
      background: #111827;
      border-bottom: 1px solid var(--card-border);
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: monospace;
    }}
    .copy-btn {{
      background: var(--card);
      border: 1px solid var(--card-border);
      color: var(--primary);
      padding: 4px 12px;
      border-radius: 5px;
      font-size: 0.78rem;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s;
    }}
    .copy-btn:hover {{ background: var(--primary); color: #000; }}
    .code-pre {{
      padding: 14px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.85rem;
      line-height: 1.55;
      color: #e2e8f0;
      overflow-x: auto;
      max-height: 400px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .meta-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 18px 0;
    }}
    .meta-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.78rem;
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
    }}
    .meta-pill strong {{ color: #f1f5f9; }}
    .badge-sol {{
      display: inline-flex;
      align-items: center;
      gap: 3px;
      font-size: 0.72rem;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
      font-weight: 600;
    }}

    .load-more-btn {{
      display: block;
      width: 220px;
      margin: 24px auto;
      padding: 12px;
      background: var(--card);
      border: 1px solid var(--card-border);
      color: var(--primary);
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      transition: all 0.15s;
    }}
    .load-more-btn:hover {{ background: var(--primary); color: #000; }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <h1>🧩 Algorithmic Pattern & Problem Explorer <span class="badge">Production Ingestion</span></h1>
    </div>
    <div class="nav-tabs">
      <button class="nav-tab active" id="tab-patterns" onclick="switchTab('patterns')">🧩 Patterns (<span id="tab-patterns-count">{len(enriched_patterns)}</span>)</button>
      <button class="nav-tab" id="tab-questions" onclick="switchTab('questions')">📚 All Questions (<span id="tab-questions-count">{len(all_questions)}</span>)</button>
      <button class="nav-tab" id="tab-outliers" onclick="switchTab('outliers')">⚡ Outliers (<span id="tab-outliers-count">{summary.get('outliers', 0)}</span>)</button>
    </div>
  </header>

  <div class="main-container">

    <!-- TAB 1: PATTERNS -->
    <div id="view-patterns">
      <div class="metrics-grid" id="metrics-bar"></div>

      <div class="toolbar">
        <div class="search-box">
          <input type="text" id="pattern-search" class="search-input" placeholder="Search patterns by label, tag, or representative problem...">
        </div>
        <select id="pattern-sort" class="filter-select" onchange="renderPatternsTable()">
          <option value="size">Sort by Size (Largest first)</option>
          <option value="id">Sort by Pattern ID</option>
        </select>
      </div>

      <table class="patterns-table">
        <thead>
          <tr>
            <th>Pattern ID</th>
            <th>Size</th>
            <th>Algorithmic Pattern & Representative</th>
            <th>Primary Tags</th>
            <th>Difficulty Distribution</th>
          </tr>
        </thead>
        <tbody id="patterns-body"></tbody>
      </table>
    </div>

    <!-- TAB 2: ALL QUESTIONS -->
    <div id="view-questions" style="display: none;">
      <div class="toolbar">
        <div class="search-box">
          <input type="text" id="question-search" class="search-input" placeholder="Search questions by title, description keyword, or tag...">
        </div>
        <select id="diff-filter" class="filter-select" onchange="resetAndRenderQuestions()">
          <option value="all">All Difficulties</option>
          <option value="easy">Easy Only</option>
          <option value="medium">Medium Only</option>
          <option value="hard">Hard Only</option>
        </select>
        <select id="pattern-filter" class="filter-select" onchange="resetAndRenderQuestions()">
          <option value="all">All Patterns & Outliers</option>
          <option value="clustered">Clustered Only</option>
          <option value="outliers">Outliers Only</option>
        </select>
        <div id="questions-counter" style="color: var(--text-muted); font-size: 0.85rem; margin-left: auto;"></div>
      </div>

      <div class="questions-grid" id="questions-grid"></div>
      <button id="load-more-btn" class="load-more-btn" onclick="loadMoreQuestions()">Load More Questions</button>
    </div>

  </div>

  <!-- PATTERN DETAIL DRAWER -->
  <div class="drawer" id="drawer">
    <div class="drawer-header">
      <div>
        <h2 id="drawer-title" style="font-size: 1.15rem; color: #fff;"></h2>
        <div id="drawer-subtitle" style="font-size: 0.85rem; color: var(--primary); margin-top: 2px;"></div>
      </div>
      <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    </div>
    <div class="drawer-body" id="drawer-body"></div>
  </div>

  <!-- QUESTION DETAIL MODAL -->
  <div class="modal-backdrop" id="modal-backdrop" onclick="closeModalOnBackdrop(event)">
    <div class="modal-dialog">
      <div class="modal-header">
        <div>
          <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 6px;">
            <span id="modal-diff" class="badge-diff"></span>
            <span id="modal-pat" class="badge-pat"></span>
          </div>
          <h2 id="modal-title" style="font-size: 1.3rem; color: #fff;"></h2>
          <div id="modal-tags" style="margin-top: 6px;"></div>
        </div>
        <button class="drawer-close" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body" id="modal-body"></div>
    </div>
  </div>

  <script>
    const patterns = {patterns_json_str};
    const questions = {questions_json_str};
    const summary = {summary_json_str};

    let activeTab = 'patterns';
    let questionPage = 1;
    const PAGE_SIZE = 48;
    let filteredQuestionsList = [];

    // Render Metrics
    document.getElementById('metrics-bar').innerHTML = `
      <div class="metric-card">
        <div class="metric-label">Total Unified Problems</div>
        <div class="metric-value">${{summary.n_records?.toLocaleString() || '3,534'}}</div>
        <div class="metric-sub">Processed & embedded</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Discovered Patterns</div>
        <div class="metric-value">${{summary.n_clusters || '102'}}</div>
        <div class="metric-sub">HDBSCAN cosine clusters</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Clustered Problems</div>
        <div class="metric-value">${{summary.clustered_records?.toLocaleString() || '2,988'}}</div>
        <div class="metric-sub">${{((summary.clustered_records / summary.n_records) * 100).toFixed(1)}}% of all problems</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Outlier Rate</div>
        <div class="metric-value">${{(summary.outlier_rate * 100)?.toFixed(1) || '15.4'}}%</div>
        <div class="metric-sub">${{summary.outliers?.toLocaleString() || '546'}} problems</div>
      </div>
    `;

    function switchTab(tab) {{
      activeTab = tab;
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

      if (tab === 'patterns') {{
        document.getElementById('tab-patterns').classList.add('active');
        document.getElementById('view-patterns').style.display = 'block';
        document.getElementById('view-questions').style.display = 'none';
        renderPatternsTable();
      }} else if (tab === 'questions') {{
        document.getElementById('tab-questions').classList.add('active');
        document.getElementById('view-patterns').style.display = 'none';
        document.getElementById('view-questions').style.display = 'block';
        document.getElementById('pattern-filter').value = 'all';
        resetAndRenderQuestions();
      }} else if (tab === 'outliers') {{
        document.getElementById('tab-outliers').classList.add('active');
        document.getElementById('view-patterns').style.display = 'none';
        document.getElementById('view-questions').style.display = 'block';
        document.getElementById('pattern-filter').value = 'outliers';
        resetAndRenderQuestions();
      }}
    }}

    /* ---------------- Patterns Table ---------------- */
    function renderPatternsTable() {{
      const query = (document.getElementById('pattern-search').value || '').toLowerCase().trim();
      const sort = document.getElementById('pattern-sort').value;

      let list = patterns.filter(p => {{
        if (!query) return true;
        const rep = questions[p.representative_id] || {{}};
        return p.label.toLowerCase().includes(query) ||
               p.pattern_id.toLowerCase().includes(query) ||
               (rep.title && rep.title.toLowerCase().includes(query)) ||
               p.top_tags.some(t => t.toLowerCase().includes(query));
      }});

      if (sort === 'size') {{
        list.sort((a, b) => b.size - a.size);
      }} else {{
        list.sort((a, b) => a.pattern_id.localeCompare(b.pattern_id, undefined, {{ numeric: true }}));
      }}

      const tbody = document.getElementById('patterns-body');
      tbody.innerHTML = list.map(p => {{
        const easy = p.difficulty.easy || 0;
        const med = p.difficulty.medium || 0;
        const hard = p.difficulty.hard || 0;
        const total = easy + med + hard || 1;
        const rep = questions[p.representative_id] || {{}};

        return `
          <tr class="pattern-row" onclick="openDrawer('${{p.pattern_id}}')">
            <td><span class="pattern-id">${{p.pattern_id}}</span></td>
            <td><span class="size-badge">${{p.size}}</span></td>
            <td>
              <div class="pattern-label">${{p.label}}</div>
              <div class="rep-name">
                <span style="color: var(--text-dim);">Representative:</span>
                <strong style="color: #fff;">${{rep.title || p.representative_id}}</strong>
                <span class="badge-diff ${{rep.difficulty || 'unknown'}}">${{rep.difficulty || 'unknown'}}</span>
              </div>
            </td>
            <td>
              ${{p.top_tags.slice(0, 4).map(t => `<span class="tag-badge">${{t}}</span>`).join('')}}
            </td>
            <td>
              <div class="diff-bar">
                <div class="diff-easy" style="width: ${{(easy/total)*100}}%"></div>
                <div class="diff-med" style="width: ${{(med/total)*100}}%"></div>
                <div class="diff-hard" style="width: ${{(hard/total)*100}}%"></div>
              </div>
              <div class="diff-text">E: ${{easy}} &nbsp; M: ${{med}} &nbsp; H: ${{hard}}</div>
            </td>
          </tr>
        `;
      }}).join('');
    }}

    /* ---------------- Pattern Drawer ---------------- */
    function openDrawer(pid) {{
      const p = patterns.find(x => x.pattern_id === pid);
      if (!p) return;

      document.getElementById('drawer-title').textContent = `${{p.pattern_id}} — ${{p.label}}`;
      document.getElementById('drawer-subtitle').textContent = `${{p.size}} clustered problems`;

      const rep = questions[p.representative_id] || {{}};

      let html = `
        <div style="background: #1e293b; border: 1px solid var(--card-border); border-radius: 10px; padding: 16px; margin-bottom: 24px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--primary); font-weight: 700;">Representative Problem</div>
            <span class="badge-diff ${{rep.difficulty || 'unknown'}}">${{rep.difficulty || 'unknown'}}</span>
          </div>
          <div style="font-size: 1.15rem; font-weight: 700; color: #fff; margin-bottom: 8px;">${{rep.title || 'Unknown'}}</div>
          <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5; margin-bottom: 12px; max-height: 80px; overflow: hidden;">
            ${{rep.description ? rep.description.substring(0, 300) + '...' : 'No description available.'}}
          </div>
          <button class="view-btn" style="padding: 6px 14px; font-size: 0.85rem;" onclick="openQuestionModal('${{p.representative_id}}')">
            📖 View Full Question Details
          </button>
        </div>

        <div style="margin-bottom: 20px;">
          <button class="view-btn" style="width: 100%; padding: 10px; text-align: center; font-size: 0.9rem;" onclick="filterByPattern('${{p.pattern_id}}')">
            🔍 Explore all ${{p.member_ids.length}} questions in Questions Tab &rarr;
          </button>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
          <h3 style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted);">
            All Member Problems (${{p.member_ids.length}})
          </h3>
          <div style="font-size: 0.8rem; color: var(--text-dim);">Click to inspect question</div>
        </div>
      `;

      html += p.member_ids.map((mid, idx) => {{
        const q = questions[mid] || {{}};
        return `
          <div class="question-card" style="margin-bottom: 8px; padding: 12px 14px; cursor: pointer;" onclick="openQuestionModal('${{mid}}')">
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px;">
              <div style="font-weight: 600; color: #fff; font-size: 0.95rem;">
                <span style="color: var(--text-dim); margin-right: 6px;">${{idx + 1}}.</span>
                ${{escapeHtml(q.title || mid)}}
              </div>
              <span class="badge-diff ${{q.difficulty || 'unknown'}}">${{q.difficulty || 'unknown'}}</span>
            </div>
            ${{q.tags && q.tags.length ? `<div style="margin-top: 6px;">${{q.tags.slice(0, 3).map(t => `<span class="tag-badge">${{escapeHtml(t)}}</span>`).join('')}}</div>` : ''}}
          </div>
        `;
      }}).join('');

      document.getElementById('drawer-body').innerHTML = html;
      document.getElementById('drawer').classList.add('open');
    }}

    function closeDrawer() {{
      document.getElementById('drawer').classList.remove('open');
    }}

    function filterByPattern(pid) {{
      closeDrawer();
      switchTab('questions');
      document.getElementById('question-search').value = pid;
      resetAndRenderQuestions();
    }}

    /* ---------------- All Questions Tab ---------------- */
    function resetAndRenderQuestions() {{
      const query = (document.getElementById('question-search').value || '').toLowerCase().trim();
      const diff = document.getElementById('diff-filter').value;
      const patFilter = document.getElementById('pattern-filter').value;

      filteredQuestionsList = Object.values(questions).filter(q => {{
        if (diff !== 'all' && q.difficulty !== diff) return false;
        if (patFilter === 'clustered' && q.pattern_id === 'Outlier') return false;
        if (patFilter === 'outliers' && q.pattern_id !== 'Outlier') return false;

        if (!query) return true;
        return (q.id && q.id.toLowerCase().includes(query)) ||
               (q.title && q.title.toLowerCase().includes(query)) ||
               (q.description && q.description.toLowerCase().includes(query)) ||
               (q.tags && q.tags.some(t => t.toLowerCase().includes(query))) ||
               (q.pattern_id && q.pattern_id.toLowerCase().includes(query)) ||
               (q.template_source_id && q.template_source_id.toLowerCase().includes(query));
      }});

      questionPage = 1;
      document.getElementById('questions-counter').textContent = `Showing ${{filteredQuestionsList.length.toLocaleString()}} questions`;
      document.getElementById('questions-grid').innerHTML = '';
      renderQuestionsPage();
    }}

    function renderQuestionsPage() {{
      const start = (questionPage - 1) * PAGE_SIZE;
      const end = start + PAGE_SIZE;
      const slice = filteredQuestionsList.slice(start, end);

      const grid = document.getElementById('questions-grid');
      const html = slice.map(q => `
        <div class="question-card" onclick="openQuestionModal('${{q.id}}')">
          <div>
            <div class="q-header">
              <div class="q-title">${{escapeHtml(q.title || 'Untitled')}}</div>
              <span class="badge-diff ${{q.difficulty}}">${{q.difficulty}}</span>
            </div>
            <div style="display: flex; gap: 6px; align-items: center; margin-bottom: 8px;">
              <span class="badge-pat ${{q.pattern_id === 'Outlier' ? 'outlier' : ''}}" onclick="event.stopPropagation(); filterByPattern('${{q.pattern_id}}')" title="Click to filter by pattern">
                ${{q.pattern_id === 'Outlier' ? '⚡ Outlier' : '🧩 ' + q.pattern_id}}
              </span>
              ${{q.canonical_solution ? '<span class="badge-sol" title="Canonical solution included">✓ Solution</span>' : ''}}
            </div>
            <div class="q-desc-snippet">${{escapeHtml(q.description || 'No description provided.')}}</div>
          </div>
          <div class="q-footer">
            <div style="flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; margin-right: 8px;">
              ${{q.tags ? q.tags.slice(0, 3).map(t => `<span class="tag-badge">${{escapeHtml(t)}}</span>`).join('') : ''}}
            </div>
            <button class="view-btn">View Details</button>
          </div>
        </div>
      `).join('');

      grid.insertAdjacentHTML('beforeend', html);

      const loadMoreBtn = document.getElementById('load-more-btn');
      if (end >= filteredQuestionsList.length) {{
        loadMoreBtn.style.display = 'none';
      }} else {{
        loadMoreBtn.style.display = 'block';
      }}
    }}

    function loadMoreQuestions() {{
      questionPage++;
      renderQuestionsPage();
    }}

    /* ---------------- Full Question Modal ---------------- */
    function openQuestionModal(qid) {{
      const q = questions[qid];
      if (!q) return;

      document.getElementById('modal-title').textContent = q.title || 'Untitled Problem';

      const diffBadge = document.getElementById('modal-diff');
      diffBadge.textContent = q.difficulty || 'unknown';
      diffBadge.className = `badge-diff ${{q.difficulty || 'unknown'}}`;

      const patBadge = document.getElementById('modal-pat');
      patBadge.textContent = q.pattern_id === 'Outlier' ? '⚡ Outlier Problem' : `🧩 Pattern: ${{q.pattern_id}}`;
      patBadge.className = `badge-pat ${{q.pattern_id === 'Outlier' ? 'outlier' : ''}}`;
      patBadge.style.cursor = 'pointer';
      patBadge.title = 'Click to view all problems in this pattern';
      patBadge.onclick = () => {{ closeModal(); filterByPattern(q.pattern_id); }};

      document.getElementById('modal-tags').innerHTML = (q.tags || []).map(t => `<span class="tag-badge">${{escapeHtml(t)}}</span>`).join('');

      let bodyHtml = `
        <div class="meta-pills">
          <div class="meta-pill"><span>Problem ID:</span> <strong>${{escapeHtml(q.id)}}</strong></div>
          ${{q.sources && q.sources.length ? `<div class="meta-pill"><span>Source:</span> <strong>${{escapeHtml(q.sources.join(', '))}}</strong></div>` : ''}}
          ${{q.template_source_id ? `<div class="meta-pill"><span>Template:</span> <strong>${{escapeHtml(q.template_source_id)}}</strong></div>` : ''}}
          ${{q.confidence !== null && q.confidence !== undefined ? `<div class="meta-pill"><span>Schema Conf:</span> <strong>${{Math.round(q.confidence * 100)}}%</strong></div>` : ''}}
          ${{q.content_hash ? `<div class="meta-pill"><span>Hash:</span> <strong>${{escapeHtml(q.content_hash.substring(0, 12))}}...</strong></div>` : ''}}
        </div>

        <div class="modal-section-title">Problem Description</div>
        <div class="desc-text">${{escapeHtml(q.description || 'No description available.')}}</div>
      `;

      // Examples
      if (q.examples && q.examples.length > 0) {{
        bodyHtml += `<div class="modal-section-title">Examples (${{q.examples.length}})</div>`;
        q.examples.forEach((ex, i) => {{
          bodyHtml += `
            <div class="example-card">
              <div class="example-title">Example ${{i + 1}}</div>
              <div class="example-field"><span class="lbl">Input:</span> <span class="val">${{escapeHtml(String(ex.input || ''))}}</span></div>
              <div class="example-field"><span class="lbl">Output:</span> <span class="val">${{escapeHtml(String(ex.output || ''))}}</span></div>
              ${{ex.explanation ? `<div class="example-field" style="margin-top: 6px;"><span class="lbl">Explanation:</span> <span style="color: #e2e8f0;">${{escapeHtml(String(ex.explanation))}}</span></div>` : ''}}
            </div>
          `;
        }});
      }}

      // Constraints
      if (q.constraints && q.constraints.length > 0) {{
        bodyHtml += `
          <div class="modal-section-title">Constraints</div>
          <ul class="constraints-list">
            ${{q.constraints.map(c => `<li><code>${{escapeHtml(c)}}</code></li>`).join('')}}
          </ul>
        `;
      }}

      // Input / Output Schemas
      if ((q.input_schema && Object.keys(q.input_schema).length > 0) || (q.output_schema && Object.keys(q.output_schema).length > 0)) {{
        bodyHtml += `
          <div class="modal-section-title">Inferred I/O Schemas</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Input Schema</div>
              <pre class="raw-json">${{escapeHtml(JSON.stringify(q.input_schema, null, 2))}}</pre>
            </div>
            <div>
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 4px;">Output Schema</div>
              <pre class="raw-json">${{escapeHtml(JSON.stringify(q.output_schema, null, 2))}}</pre>
            </div>
          </div>
        `;
      }}

      // Canonical Solution
      if (q.canonical_solution && q.canonical_solution.trim()) {{
        bodyHtml += `
          <div class="modal-section-title">Canonical Solution</div>
          <div class="code-container">
            <div class="code-toolbar">
              <span>Python Solution Code</span>
              <button class="copy-btn" id="copy-sol-btn" onclick="copySolutionText()">📋 Copy Solution</button>
            </div>
            <pre class="code-pre" id="solution-code-block">${{escapeHtml(q.canonical_solution)}}</pre>
          </div>
        `;
      }}

      // Complete Raw JSON Toggle
      bodyHtml += `
        <div class="modal-section-title">Raw Problem Record</div>
        <details>
          <summary style="font-size: 0.85rem; color: var(--primary); cursor: pointer; margin-bottom: 8px;">View Full JSON Record</summary>
          <pre class="raw-json">${{escapeHtml(JSON.stringify(q, null, 2))}}</pre>
        </details>
      `;

      document.getElementById('modal-body').innerHTML = bodyHtml;
      document.getElementById('modal-backdrop').classList.add('open');
    }}

    function copySolutionText() {{
      const codeEl = document.getElementById('solution-code-block');
      if (!codeEl) return;
      navigator.clipboard.writeText(codeEl.textContent).then(() => {{
        const btn = document.getElementById('copy-sol-btn');
        if (btn) {{
          btn.textContent = '✓ Copied!';
          setTimeout(() => {{ btn.textContent = '📋 Copy Solution'; }}, 2000);
        }}
      }}).catch(() => {{
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = codeEl.textContent;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        const btn = document.getElementById('copy-sol-btn');
        if (btn) {{
          btn.textContent = '✓ Copied!';
          setTimeout(() => {{ btn.textContent = '📋 Copy Solution'; }}, 2000);
        }}
      }});
    }}

    function closeModal() {{
      document.getElementById('modal-backdrop').classList.remove('open');
    }}

    function closeModalOnBackdrop(event) {{
      if (event.target.id === 'modal-backdrop') {{
        closeModal();
      }}
    }}

    function escapeHtml(str) {{
      return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    // Close on Escape key
    window.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') {{
        closeModal();
        closeDrawer();
      }}
    }});

    // Live search listeners
    document.getElementById('pattern-search').addEventListener('input', renderPatternsTable);
    document.getElementById('question-search').addEventListener('input', resetAndRenderQuestions);

    // Initial render
    renderPatternsTable();
  </script>
</body>
</html>
"""
    out.write_text(html_content, encoding="utf-8")
    return out
