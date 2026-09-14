"""HTML dashboard generator for template re-authoring pipeline results.

Mirrors `pattern_mining/viewer.py`'s style/theme (dark UI, metric cards,
searchable tables, detail modals) so the two dashboards feel like one
product, but surfaces transform-pipeline-specific data: accepted
templates with their rewritten text and variants, rejected records with
failure reasons, and run-level metrics (acceptance rate, timing,
needs-review breakdown).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional


def load_run_data(
    db_path: str | Path,
    unified_json_path: str | Path,
    run_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Loads templates + variants from the DB and joins each template
    back to its canonical record (title, difficulty, tags, source) from
    unified_dataset.json for display context. `run_metadata` is an
    optional dict (typically the JSON blob written by a batch run
    script) merged in as top-level run stats (elapsed time, sample size,
    rejections list) since that data doesn't live in the DB.
    """
    with open(unified_json_path, "r", encoding="utf-8") as f:
        unified = json.load(f)
    by_id = {r["id"]: r for r in unified}

    # Ensure the templates/variants tables exist even for a run that
    # rejected every record (so no template was ever written) — without
    # this, querying a freshly-created or migration-less DB path raises
    # sqlite3.OperationalError instead of returning an empty result set.
    from .storage import migrate_db

    migrate_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    templates = []
    for row in conn.execute("SELECT * FROM templates ORDER BY created_at"):
        canonical = by_id.get(row["canonical_problem_id"], {})
        variants = [
            dict(v) for v in conn.execute(
                "SELECT * FROM variants WHERE template_id = ?", (row["template_id"],)
            )
        ]
        for v in variants:
            v["args"] = json.loads(v["args"]) if v["args"] else {}
            v["canonical_output"] = json.loads(v["canonical_output"]) if v["canonical_output"] else None

        templates.append(
            {
                "template_id": row["template_id"],
                "canonical_problem_id": row["canonical_problem_id"],
                "pattern_id": row["pattern_id"],
                "title": row["title"],
                "description": row["description"],
                "input_schema": json.loads(row["input_schema"]) if row["input_schema"] else {},
                "output_schema": json.loads(row["output_schema"]) if row["output_schema"] else {},
                "constraints": json.loads(row["constraints"]) if row["constraints"] else [],
                "tie_breaker": row["tie_breaker"],
                "transform_seed": row["transform_seed"],
                "transform_model": row["transform_model"],
                "signature_hash": row["signature_hash"],
                "needs_review": bool(row["needs_review"]),
                "review_reasons": json.loads(row["review_reasons"]) if row["review_reasons"] else [],
                "provenance": json.loads(row["provenance"]) if row["provenance"] else {},
                "created_at": row["created_at"],
                "variants": variants,
                "canonical_title": canonical.get("title", "Unknown"),
                "canonical_difficulty": canonical.get("difficulty", "unknown"),
                "canonical_tags": canonical.get("tags", []),
                "canonical_sources": canonical.get("source_list", []),
                "canonical_solution_code": (canonical.get("canonical_solution") or {}).get("code", ""),
            }
        )

    conn.close()

    rejections = []
    if run_metadata:
        for rej in run_metadata.get("rejections", []):
            canonical = by_id.get(rej["canonical_problem_id"], {})
            reasons = rej["reason"] if isinstance(rej["reason"], list) else [rej["reason"]]
            rejections.append(
                {
                    "canonical_problem_id": rej["canonical_problem_id"],
                    "canonical_title": canonical.get("title", "Unknown"),
                    "canonical_difficulty": canonical.get("difficulty", "unknown"),
                    "reasons": reasons,
                }
            )

    return {
        "templates": templates,
        "rejections": rejections,
        "run_metadata": run_metadata or {},
    }


def generate_transform_dashboard(
    run_data: dict[str, Any],
    output_path: str | Path = "data/output/transform_dashboard.html",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    templates = run_data["templates"]
    rejections = run_data["rejections"]
    meta = run_data["run_metadata"]
    metrics = meta.get("metrics", {})

    templates_json_str = json.dumps(templates, default=str)
    rejections_json_str = json.dumps(rejections, default=str)
    metrics_json_str = json.dumps(metrics)
    meta_json_str = json.dumps(
        {
            "sample_size": meta.get("sample_size"),
            "total_eligible_candidates": meta.get("total_eligible_candidates"),
            "total_dataset_records": meta.get("total_dataset_records"),
            "elapsed_seconds": meta.get("elapsed_seconds"),
        }
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Template Re-authoring Pipeline Results</title>
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
      --easy: #10b981;
      --easy-bg: rgba(16, 185, 129, 0.15);
      --medium: #f59e0b;
      --medium-bg: rgba(245, 158, 11, 0.15);
      --hard: #ef4444;
      --hard-bg: rgba(239, 68, 68, 0.15);
      --accept: #10b981;
      --accept-bg: rgba(16, 185, 129, 0.15);
      --reject: #ef4444;
      --reject-bg: rgba(239, 68, 68, 0.15);
      --review: #f59e0b;
      --review-bg: rgba(245, 158, 11, 0.15);
      --code-bg: #030712;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
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
      background: transparent; border: 1px solid transparent; color: var(--text-muted);
      padding: 8px 16px; border-radius: 8px; font-size: 0.9rem; font-weight: 500; cursor: pointer; transition: all 0.15s;
    }}
    .nav-tab:hover {{ color: var(--text); background: rgba(255, 255, 255, 0.05); }}
    .nav-tab.active {{ color: #000; background: var(--primary); font-weight: 600; border-color: var(--primary); }}

    .main-container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}

    .metrics-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--surface); border: 1px solid var(--card-border); border-radius: 12px;
      padding: 16px 20px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    .metric-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); font-weight: 600; }}
    .metric-value {{ font-size: 1.7rem; font-weight: 700; color: #fff; margin-top: 4px; }}
    .metric-sub {{ font-size: 0.78rem; color: var(--primary); margin-top: 2px; }}
    .metric-card.accept .metric-value {{ color: var(--accept); }}
    .metric-card.reject .metric-value {{ color: var(--reject); }}
    .metric-card.review .metric-value {{ color: var(--review); }}

    .section-title {{
      font-size: 0.95rem; font-weight: 700; color: #fff; margin: 28px 0 14px 0;
      display: flex; align-items: center; gap: 8px;
    }}

    .callout {{
      background: var(--surface); border: 1px solid var(--card-border); border-left: 3px solid var(--primary);
      border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; font-size: 0.88rem; color: var(--text-muted); line-height: 1.6;
    }}
    .callout strong {{ color: #fff; }}

    .toolbar {{
      background: var(--surface); border: 1px solid var(--card-border); border-radius: 12px;
      padding: 14px 18px; display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 20px;
    }}
    .search-box {{ flex: 1; min-width: 260px; }}
    .search-input {{
      width: 100%; background: var(--card); border: 1px solid var(--card-border); color: #fff;
      padding: 10px 14px; border-radius: 8px; font-size: 0.95rem;
    }}
    .search-input:focus {{ outline: none; border-color: var(--primary); }}
    .filter-select {{
      background: var(--card); border: 1px solid var(--card-border); color: #fff;
      padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; cursor: pointer;
    }}

    .bar-chart {{ display: flex; flex-direction: column; gap: 10px; }}
    .bar-row {{ display: flex; align-items: center; gap: 12px; }}
    .bar-label {{ width: 340px; font-size: 0.82rem; color: var(--text-muted); flex-shrink: 0; }}
    .bar-track {{ flex: 1; height: 22px; background: var(--card); border-radius: 6px; overflow: hidden; position: relative; }}
    .bar-fill {{ height: 100%; background: var(--primary); border-radius: 6px; }}
    .bar-fill.reject {{ background: var(--reject); }}
    .bar-fill.review {{ background: var(--review); }}
    .bar-count {{ width: 40px; text-align: right; font-size: 0.85rem; font-weight: 700; color: #fff; }}

    .item-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }}
    .item-card {{
      background: var(--surface); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px;
      cursor: pointer; transition: all 0.15s; border-left: 3px solid var(--card-border);
    }}
    .item-card:hover {{ border-color: var(--primary); transform: translateY(-2px); }}
    .item-card.accepted {{ border-left-color: var(--accept); }}
    .item-card.rejected {{ border-left-color: var(--reject); }}
    .item-card.needs-review {{ border-left-color: var(--review); }}
    .item-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 8px; }}
    .item-title {{ font-size: 1.0rem; font-weight: 600; color: #fff; line-height: 1.35; }}
    .item-sub {{ font-size: 0.8rem; color: var(--text-dim); margin-bottom: 8px; }}
    .item-desc {{ color: var(--text-muted); font-size: 0.83rem; line-height: 1.4; margin: 8px 0; max-height: 3.8em; overflow: hidden; }}
    .badge-diff {{
      font-size: 0.7rem; padding: 3px 8px; border-radius: 6px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.04em; white-space: nowrap;
    }}
    .badge-diff.easy {{ background: var(--easy-bg); color: var(--easy); border: 1px solid rgba(16,185,129,0.3); }}
    .badge-diff.medium {{ background: var(--medium-bg); color: var(--medium); border: 1px solid rgba(245,158,11,0.3); }}
    .badge-diff.hard {{ background: var(--hard-bg); color: var(--hard); border: 1px solid rgba(239,68,68,0.3); }}
    .badge-diff.unknown {{ background: rgba(156,163,175,0.15); color: #9ca3af; border: 1px solid rgba(156,163,175,0.3); }}
    .badge-status {{
      font-size: 0.7rem; padding: 3px 9px; border-radius: 6px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .badge-status.accept {{ background: var(--accept-bg); color: var(--accept); border: 1px solid rgba(16,185,129,0.3); }}
    .badge-status.reject {{ background: var(--reject-bg); color: var(--reject); border: 1px solid rgba(239,68,68,0.3); }}
    .badge-status.review {{ background: var(--review-bg); color: var(--review); border: 1px solid rgba(245,158,11,0.3); }}
    .tag-badge {{
      display: inline-block; background: var(--card); color: #d1d5db; font-size: 0.72rem;
      padding: 3px 8px; border-radius: 6px; margin: 2px 4px 2px 0; border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    .item-footer {{ margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.78rem; color: var(--text-dim); }}

    /* Modal */
    .modal-backdrop {{
      position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0, 0, 0, 0.75);
      backdrop-filter: blur(4px); z-index: 200; display: none; align-items: center; justify-content: center; padding: 24px;
    }}
    .modal-backdrop.open {{ display: flex; }}
    .modal-dialog {{
      background: #111827; border: 1px solid var(--card-border); border-radius: 16px; width: 920px; max-width: 95vw;
      max-height: 90vh; display: flex; flex-direction: column; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8);
    }}
    .modal-header {{
      padding: 20px 28px; border-bottom: 1px solid var(--card-border); background: #1f2937;
      border-top-left-radius: 16px; border-top-right-radius: 16px; display: flex; justify-content: space-between; align-items: flex-start;
    }}
    .modal-close {{ background: none; border: none; color: var(--text-muted); font-size: 1.8rem; cursor: pointer; }}
    .modal-close:hover {{ color: #fff; }}
    .modal-body {{ padding: 28px; overflow-y: auto; flex: 1; }}
    .modal-section-title {{
      font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--primary); font-weight: 700;
      margin: 20px 0 8px 0; border-bottom: 1px solid var(--card-border); padding-bottom: 4px;
    }}
    .modal-section-title:first-child {{ margin-top: 0; }}
    .desc-text {{ font-size: 0.95rem; color: #e5e7eb; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }}
    .code-container {{
      position: relative; background: var(--code-bg); border: 1px solid var(--card-border); border-radius: 8px;
      overflow: hidden; margin-top: 8px;
    }}
    .code-toolbar {{
      display: flex; justify-content: space-between; align-items: center; padding: 8px 14px; background: #111827;
      border-bottom: 1px solid var(--card-border); font-size: 0.75rem; color: var(--text-muted); font-family: monospace;
    }}
    .code-pre {{
      padding: 14px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.83rem;
      line-height: 1.55; color: #e2e8f0; overflow-x: auto; max-height: 340px; white-space: pre-wrap; word-break: break-word;
    }}
    .example-card {{
      background: var(--code-bg); border: 1px solid #1f2937; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.85rem;
    }}
    .example-field span.lbl {{ color: var(--text-muted); }}
    .example-field span.val {{ color: #a5f3fc; }}
    .reasons-list {{ padding-left: 18px; color: #fca5a5; font-size: 0.88rem; line-height: 1.7; }}
    .reasons-list.review {{ color: #fcd34d; }}
    .meta-pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px 0; }}
    .meta-pill {{
      display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 6px; font-size: 0.76rem;
      background: #1e293b; border: 1px solid var(--card-border); color: var(--text-muted);
    }}
    .meta-pill strong {{ color: #f1f5f9; }}
    .variant-row {{
      display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: var(--code-bg);
      border: 1px solid #1f2937; border-radius: 6px; margin-bottom: 6px; font-family: monospace; font-size: 0.8rem;
    }}
    .load-more-btn {{
      display: block; width: 220px; margin: 20px auto; padding: 12px; background: var(--card); border: 1px solid var(--card-border);
      color: var(--primary); border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; text-align: center;
    }}
    .load-more-btn:hover {{ background: var(--primary); color: #000; }}
    .empty-state {{ text-align: center; padding: 60px 20px; color: var(--text-dim); }}
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <h1>🔁 Template Re-authoring Results <span class="badge">Dry-Run / Mock LLM</span></h1>
    </div>
    <div class="nav-tabs">
      <button class="nav-tab active" id="tab-overview" onclick="switchTab('overview')">📊 Overview</button>
      <button class="nav-tab" id="tab-accepted" onclick="switchTab('accepted')">✅ Accepted (<span id="count-accepted">{sum(1 for t in templates if not t["needs_review"])}</span>)</button>
      <button class="nav-tab" id="tab-review" onclick="switchTab('review')">⚠️ Needs Review (<span id="count-review">{sum(1 for t in templates if t["needs_review"])}</span>)</button>
      <button class="nav-tab" id="tab-rejected" onclick="switchTab('rejected')">❌ Rejected (<span id="count-rejected">{len(rejections)}</span>)</button>
    </div>
  </header>

  <div class="main-container">

    <div id="view-overview">
      <div class="callout">
        <strong>What this run is:</strong> a dry run of the <code>transform/</code> re-authoring pipeline against
        real canonical problems pulled from <code>unified_dataset.json</code>, using a <strong>mocked LLM</strong>
        that performs a passthrough rewrite (no actual domain-swap/rephrasing) — no local model (Ollama/Mistral)
        was installed in this environment. Every validation result below is real: the canonical solver was
        actually executed in the sandbox for every sample test and every generated variant. This measures the
        pipeline's plumbing and the sandbox's solution-execution coverage, not rewrite quality — that requires a
        real LLM.
      </div>

      <div class="metrics-grid" id="metrics-bar"></div>

      <div class="section-title">📉 Rejection Reasons</div>
      <div class="bar-chart" id="rejection-chart"></div>

      <div class="section-title">📋 Needs-Review Reasons (accepted, but flagged)</div>
      <div class="bar-chart" id="review-chart"></div>

      <div class="section-title">🏷️ Accepted Templates by Difficulty</div>
      <div class="bar-chart" id="difficulty-chart"></div>
    </div>

    <div id="view-accepted" style="display: none;">
      <div class="toolbar">
        <div class="search-box">
          <input type="text" id="accepted-search" class="search-input" placeholder="Search accepted templates by title or tag...">
        </div>
      </div>
      <div class="item-grid" id="accepted-grid"></div>
      <button id="accepted-load-more" class="load-more-btn" onclick="loadMore('accepted')">Load More</button>
    </div>

    <div id="view-review" style="display: none;">
      <div class="toolbar">
        <div class="search-box">
          <input type="text" id="review-search" class="search-input" placeholder="Search needs-review templates...">
        </div>
      </div>
      <div class="item-grid" id="review-grid"></div>
      <button id="review-load-more" class="load-more-btn" onclick="loadMore('review')">Load More</button>
    </div>

    <div id="view-rejected" style="display: none;">
      <div class="toolbar">
        <div class="search-box">
          <input type="text" id="rejected-search" class="search-input" placeholder="Search rejected records by title or reason...">
        </div>
      </div>
      <div class="item-grid" id="rejected-grid"></div>
      <button id="rejected-load-more" class="load-more-btn" onclick="loadMore('rejected')">Load More</button>
    </div>

  </div>

  <div class="modal-backdrop" id="modal-backdrop" onclick="closeModalOnBackdrop(event)">
    <div class="modal-dialog">
      <div class="modal-header">
        <div>
          <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 6px;" id="modal-badges"></div>
          <h2 id="modal-title" style="font-size: 1.25rem; color: #fff;"></h2>
          <div id="modal-tags" style="margin-top: 6px;"></div>
        </div>
        <button class="modal-close" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body" id="modal-body"></div>
    </div>
  </div>

  <script>
    const templates = {templates_json_str};
    const rejections = {rejections_json_str};
    const metrics = {metrics_json_str};
    const runMeta = {meta_json_str};

    const accepted = templates.filter(t => !t.needs_review);
    const needsReview = templates.filter(t => t.needs_review);

    const PAGE_SIZE = 24;
    const pages = {{ accepted: 1, review: 1, rejected: 1 }};
    let filtered = {{ accepted: accepted, review: needsReview, rejected: rejections }};

    function escapeHtml(str) {{
      if (str === null || str === undefined) return '';
      return String(str)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }}

    function switchTab(tab) {{
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.getElementById('tab-' + tab).classList.add('active');
      ['overview', 'accepted', 'review', 'rejected'].forEach(v => {{
        document.getElementById('view-' + v).style.display = (v === tab) ? 'block' : 'none';
      }});
      if (tab === 'accepted') resetAndRender('accepted');
      if (tab === 'review') resetAndRender('review');
      if (tab === 'rejected') resetAndRender('rejected');
    }}

    // ---- Overview metrics ----
    document.getElementById('metrics-bar').innerHTML = `
      <div class="metric-card">
        <div class="metric-label">Records Considered</div>
        <div class="metric-value">${{(metrics.records_considered || 0).toLocaleString()}}</div>
        <div class="metric-sub">of ${{(runMeta.total_eligible_candidates || 0).toLocaleString()}} eligible (Python + solution + examples)</div>
      </div>
      <div class="metric-card accept">
        <div class="metric-label">Accepted</div>
        <div class="metric-value">${{(metrics.accepted || 0).toLocaleString()}}</div>
        <div class="metric-sub">${{((metrics.acceptance_rate || 0) * 100).toFixed(1)}}% acceptance rate</div>
      </div>
      <div class="metric-card reject">
        <div class="metric-label">Rejected</div>
        <div class="metric-value">${{((metrics.rejected_invalid_json || 0) + (metrics.rejected_validation_failed || 0)).toLocaleString()}}</div>
        <div class="metric-sub">${{metrics.rejected_validation_failed || 0}} failed validation, ${{metrics.rejected_invalid_json || 0}} bad JSON</div>
      </div>
      <div class="metric-card review">
        <div class="metric-label">Needs Review</div>
        <div class="metric-value">${{(metrics.needs_review || 0).toLocaleString()}}</div>
        <div class="metric-sub">accepted, but flagged for a human look</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Variants Generated</div>
        <div class="metric-value">${{(metrics.total_variants_generated || 0).toLocaleString()}}</div>
        <div class="metric-sub">randomized inputs, canonical solver re-run on each</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Run Time</div>
        <div class="metric-value">${{runMeta.elapsed_seconds ? runMeta.elapsed_seconds.toFixed(1) + 's' : '—'}}</div>
        <div class="metric-sub">${{runMeta.elapsed_seconds && runMeta.sample_size ? (runMeta.elapsed_seconds / runMeta.sample_size).toFixed(2) + 's / record avg' : ''}}</div>
      </div>
    `;

    // ---- Rejection reason chart ----
    function categorizeReason(r) {{
      if (r.includes('sample_public_tests did not reproduce')) return 'Sample tests did not reproduce under canonical solver';
      if (r.includes('no detectable entry point')) return 'Canonical solver has no detectable entry point';
      if (r.includes('no python canonical solution')) return 'No Python canonical solution available';
      if (r.includes('argument count mismatch')) return 'Argument count mismatch (rewrite added/removed a field)';
      if (r.includes('zero sample_public_tests')) return 'Template declared zero sample tests';
      if (r.includes('data_unavailable')) return 'data_unavailable (no solution to validate against)';
      return r.slice(0, 70);
    }}

    function renderBarChart(containerId, counts, barClass) {{
      const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      const max = entries.length ? entries[0][1] : 1;
      const el = document.getElementById(containerId);
      if (!entries.length) {{
        el.innerHTML = '<div class="empty-state">No data.</div>';
        return;
      }}
      el.innerHTML = entries.map(([label, count]) => `
        <div class="bar-row">
          <div class="bar-label" title="${{escapeHtml(label)}}">${{escapeHtml(label)}}</div>
          <div class="bar-track"><div class="bar-fill ${{barClass}}" style="width: ${{(count / max) * 100}}%"></div></div>
          <div class="bar-count">${{count}}</div>
        </div>
      `).join('');
    }}

    const rejectionCounts = {{}};
    rejections.forEach(r => {{
      r.reasons.forEach(reason => {{
        const key = categorizeReason(reason);
        rejectionCounts[key] = (rejectionCounts[key] || 0) + 1;
      }});
    }});
    renderBarChart('rejection-chart', rejectionCounts, 'reject');

    const reviewCounts = {{}};
    needsReview.forEach(t => {{
      t.review_reasons.forEach(reason => {{
        let key;
        if (reason.includes('tie_breaker')) key = 'No tie_breaker declared';
        else if (reason.includes('output_schema shape changed')) key = 'output_schema shape changed vs canonical';
        else if (reason.includes('generated variants failed')) key = 'Some generated variants failed on canonical solver';
        else if (reason.includes('skipped:')) key = 'Variant generation skipped (unsupported schema type)';
        else key = reason.slice(0, 70);
        reviewCounts[key] = (reviewCounts[key] || 0) + 1;
      }});
    }});
    renderBarChart('review-chart', reviewCounts, 'review');

    const difficultyCounts = {{}};
    accepted.concat(needsReview).forEach(t => {{
      const d = t.canonical_difficulty || 'unknown';
      difficultyCounts[d] = (difficultyCounts[d] || 0) + 1;
    }});
    renderBarChart('difficulty-chart', difficultyCounts, '');

    // ---- List rendering (accepted / review / rejected tabs) ----
    function resetAndRender(kind) {{
      const searchId = kind + '-search';
      const searchEl = document.getElementById(searchId);
      const query = searchEl ? (searchEl.value || '').toLowerCase().trim() : '';

      const source = kind === 'accepted' ? accepted : (kind === 'review' ? needsReview : rejections);
      filtered[kind] = source.filter(item => {{
        if (!query) return true;
        if (kind === 'rejected') {{
          return (item.canonical_title || '').toLowerCase().includes(query) ||
                 item.reasons.some(r => r.toLowerCase().includes(query));
        }}
        return (item.title || '').toLowerCase().includes(query) ||
               (item.canonical_title || '').toLowerCase().includes(query) ||
               (item.canonical_tags || []).some(t => t.toLowerCase().includes(query));
      }});
      pages[kind] = 1;
      document.getElementById(kind + '-grid').innerHTML = '';
      renderPage(kind);
    }}

    function renderPage(kind) {{
      const start = (pages[kind] - 1) * PAGE_SIZE;
      const end = start + PAGE_SIZE;
      const slice = filtered[kind].slice(start, end);
      const grid = document.getElementById(kind + '-grid');

      if (filtered[kind].length === 0) {{
        grid.innerHTML = '<div class="empty-state">No items match.</div>';
        document.getElementById(kind + '-load-more').style.display = 'none';
        return;
      }}

      let html = '';
      if (kind === 'rejected') {{
        html = slice.map((r, i) => `
          <div class="item-card rejected" onclick="openRejectedModal(${{templates.length + i}})" data-idx="${{rejections.indexOf(r)}}" onclick="openRejectedModalByIdx(${{rejections.indexOf(r)}})">
            <div class="item-header">
              <div class="item-title">${{escapeHtml(r.canonical_title)}}</div>
              <span class="badge-diff ${{r.canonical_difficulty}}">${{r.canonical_difficulty}}</span>
            </div>
            <div class="item-sub">${{escapeHtml(r.canonical_problem_id.slice(0, 12))}}...</div>
            <div class="item-desc" style="color: #fca5a5;">${{escapeHtml(r.reasons[0] || '')}}</div>
            <div class="item-footer">${{r.reasons.length}} reason(s) — click to view</div>
          </div>
        `).join('');
      }} else {{
        html = slice.map(t => `
          <div class="item-card ${{t.needs_review ? 'needs-review' : 'accepted'}}" onclick="openTemplateModal('${{t.template_id}}')">
            <div class="item-header">
              <div class="item-title">${{escapeHtml(t.title)}}</div>
              <span class="badge-status ${{t.needs_review ? 'review' : 'accept'}}">${{t.needs_review ? 'Review' : 'Accepted'}}</span>
            </div>
            <div class="item-sub">Rewritten from: <strong style="color:#d1d5db;">${{escapeHtml(t.canonical_title)}}</strong> <span class="badge-diff ${{t.canonical_difficulty}}">${{t.canonical_difficulty}}</span></div>
            <div class="item-desc">${{escapeHtml(t.description || '')}}</div>
            <div class="item-footer">${{t.variants.length}} variant(s) validated &middot; sig: ${{(t.signature_hash || '').slice(0, 10)}}...</div>
          </div>
        `).join('');
      }}
      grid.insertAdjacentHTML('beforeend', html);

      const btn = document.getElementById(kind + '-load-more');
      btn.style.display = (end >= filtered[kind].length) ? 'none' : 'block';
    }}

    function loadMore(kind) {{
      pages[kind]++;
      renderPage(kind);
    }}

    // ---- Modals ----
    function openTemplateModal(templateId) {{
      const t = templates.find(x => x.template_id === templateId);
      if (!t) return;

      document.getElementById('modal-title').textContent = t.title;
      document.getElementById('modal-badges').innerHTML = `
        <span class="badge-status ${{t.needs_review ? 'review' : 'accept'}}">${{t.needs_review ? 'Needs Review' : 'Accepted'}}</span>
        <span class="badge-diff ${{t.canonical_difficulty}}">${{t.canonical_difficulty}}</span>
      `;
      document.getElementById('modal-tags').innerHTML = (t.canonical_tags || []).map(tag => `<span class="tag-badge">${{escapeHtml(tag)}}</span>`).join('');

      let body = `
        <div class="meta-pills">
          <div class="meta-pill"><span>Template ID:</span> <strong>${{escapeHtml(t.template_id.slice(0,12))}}...</strong></div>
          <div class="meta-pill"><span>Canonical Problem:</span> <strong>${{escapeHtml(t.canonical_title)}}</strong></div>
          <div class="meta-pill"><span>Transform Model:</span> <strong>${{escapeHtml(t.transform_model)}}</strong></div>
          <div class="meta-pill"><span>Signature:</span> <strong>${{escapeHtml((t.signature_hash || '').slice(0,14))}}...</strong></div>
        </div>

        <div class="modal-section-title">Rewritten Description</div>
        <div class="desc-text">${{escapeHtml(t.description)}}</div>
      `;

      if (t.tie_breaker) {{
        body += `<div class="modal-section-title">Tie-Breaker</div><div class="desc-text">${{escapeHtml(t.tie_breaker)}}</div>`;
      }}

      if (t.constraints && t.constraints.length) {{
        body += `<div class="modal-section-title">Constraints</div><ul class="reasons-list" style="color:#d1d5db;">${{t.constraints.map(c => `<li>${{escapeHtml(c)}}</li>`).join('')}}</ul>`;
      }}

      if (t.needs_review && t.review_reasons.length) {{
        body += `<div class="modal-section-title">⚠️ Review Reasons</div><ul class="reasons-list review">${{t.review_reasons.map(r => `<li>${{escapeHtml(r)}}</li>`).join('')}}</ul>`;
      }}

      if (t.canonical_solution_code) {{
        body += `
          <div class="modal-section-title">Canonical Solver (validated against, not the LLM's output)</div>
          <div class="code-container">
            <div class="code-toolbar"><span>Python</span></div>
            <pre class="code-pre">${{escapeHtml(t.canonical_solution_code)}}</pre>
          </div>
        `;
      }}

      if (t.variants && t.variants.length) {{
        body += `<div class="modal-section-title">Validated Variants (${{t.variants.length}})</div>`;
        t.variants.slice(0, 10).forEach((v, i) => {{
          body += `
            <div class="variant-row">
              <span>#${{i+1}} args=${{escapeHtml(JSON.stringify(v.args))}}</span>
              <span style="color:#34d399;">→ ${{escapeHtml(JSON.stringify(v.canonical_output))}}</span>
            </div>
          `;
        }});
        if (t.variants.length > 10) {{
          body += `<div style="color:var(--text-dim); font-size:0.8rem;">...and ${{t.variants.length - 10}} more</div>`;
        }}
      }}

      body += `
        <div class="modal-section-title">Raw Template Record</div>
        <details>
          <summary style="font-size: 0.85rem; color: var(--primary); cursor: pointer; margin-bottom: 8px;">View Full JSON</summary>
          <pre class="code-pre">${{escapeHtml(JSON.stringify(t, null, 2))}}</pre>
        </details>
      `;

      document.getElementById('modal-body').innerHTML = body;
      document.getElementById('modal-backdrop').classList.add('open');
    }}

    function openRejectedModalByIdx(idx) {{
      const r = rejections[idx];
      if (!r) return;

      document.getElementById('modal-title').textContent = r.canonical_title;
      document.getElementById('modal-badges').innerHTML = `
        <span class="badge-status reject">Rejected</span>
        <span class="badge-diff ${{r.canonical_difficulty}}">${{r.canonical_difficulty}}</span>
      `;
      document.getElementById('modal-tags').innerHTML = '';

      const body = `
        <div class="meta-pills">
          <div class="meta-pill"><span>Canonical Problem ID:</span> <strong>${{escapeHtml(r.canonical_problem_id)}}</strong></div>
        </div>
        <div class="modal-section-title">❌ Rejection Reasons</div>
        <ul class="reasons-list">${{r.reasons.map(reason => `<li>${{escapeHtml(reason)}}</li>`).join('')}}</ul>
      `;
      document.getElementById('modal-body').innerHTML = body;
      document.getElementById('modal-backdrop').classList.add('open');
    }}

    function closeModal() {{ document.getElementById('modal-backdrop').classList.remove('open'); }}
    function closeModalOnBackdrop(event) {{ if (event.target.id === 'modal-backdrop') closeModal(); }}
    window.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') closeModal(); }});

    document.getElementById('accepted-search').addEventListener('input', () => resetAndRender('accepted'));
    document.getElementById('review-search').addEventListener('input', () => resetAndRender('review'));
    document.getElementById('rejected-search').addEventListener('input', () => resetAndRender('rejected'));
  </script>
</body>
</html>
"""
    out.write_text(html_content, encoding="utf-8")
    return out
