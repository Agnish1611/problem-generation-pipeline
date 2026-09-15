# Code Royale — Problem-Generation Pipeline Overview

> **Single-page technical reference** for the full pipeline: what each layer does, how they connect, what the real performance numbers are, and where the upgrade seams are.

---

## Architecture at a Glance

```
Raw Sources (JSONL / CSV / JSON / Markdown)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  ingestion/  — Source Adapters → Normalizer →   │
│  Schema Extractor → Fingerprinting →            │
│  License Tracker → De-duplication →             │
│  Validation → Storage (SQLite)                  │
│                  ↓                              │
│         unified_dataset.json                    │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  pattern_mining/  — Embeddings (sentence-       │
│  transformers) → UMAP + HDBSCAN Clustering →   │
│  Cluster Summarization → SQLite + patterns.json │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  transform/  — Spec Builder → LLM Prompt →      │
│  JSON Parse/Repair → Validator (runs canonical  │
│  solver via sandbox/) → SQLite templates table  │
└─────────────────────────────────────────────────┘
```

---

## Layer 1 — Ingestion (`ingestion/`)

### Purpose
Turns raw source dumps into a single clean, versioned, auditable `CanonicalRecord` dataset.

### Data Sources

| Source | Kind | License | Notes |
|--------|------|---------|-------|
| `data/raw/leetcode_doocs/` | Markdown repo | CC-BY-SA-4.0 | Primary/canonical; wins dedup ties |
| `data/raw/leetcode_problems.json` | JSON | unknown | Backfills metadata |
| `data/raw/leetcode_dataset_train.jsonl` | JSONL | educational | ~94 MB; backfills solutions |
| `data/raw/leetcode_dataset_test.jsonl` | JSONL | educational | ~7.5 MB test split |
| `data/raw/leetcode_kaggle.csv` | CSV | unknown | Kaggle scraped snapshot |

### Pipeline Stages

| Stage | Module | Output |
|-------|--------|--------|
| Source Adapter | `adapters.py` | `RawProblem` (provenance attached) |
| Normalizer | `normalizer.py` | `CanonicalRecord` (clean text, canonical tags/difficulty) |
| Schema Extractor | `schema_extractor.py` | `input_schema` / `output_schema` + confidence score |
| Fingerprinting | `fingerprint.py` | `content_hash`, `nl_embedding` (TF-IDF), `ast_signature` |
| License Tracker | `license_tracker.py` | `requires_legal_review`, `public_allowed` flags |
| De-duplication | `dedup.py` | Survivors list; duplicates marked `deduplicated_into` |
| Validation | `validation.py` | `needs_review` flags + reasons |
| Storage | `storage.py` | SQLite upsert → `ingestion.db`; JSON export → `unified_dataset.json` |

### De-duplication Strategy (4-stage, cheapest first)

1. **Exact normalized-title hash** — O(N) bucket grouping
2. **Token-Jaccard** on title + first 50 description tokens (thresholds: 0.8 title, 0.6 desc)
3. **Cosine similarity ≥ 0.90** on TF-IDF description embeddings
4. **AST signature exact match** on canonical solution code

**Sequel-problem guard**: titles differing only by a roman-numeral/Part-N suffix (e.g. "Two Sum" vs "Two Sum II") are always treated as distinct, regardless of description similarity.

### Performance Metrics (last full ingest)

| Metric | Value |
|--------|-------|
| Canonical records stored | **4,387** |
| Records with solution code | **4,236** (96.6%) |
| Premium problems with no data | 29 (`data_unavailable=True`) |
| Bogus `code_ast` merges (post-fix) | **9** (was 255 before comment-stripping bug fix) |
| `needs_review` rate | ~100% (all CC-BY-SA-4.0 doocs records require legal review by policy) |

### Key Design Decisions

- **Idempotency**: re-running on the same source file bumps `version` but never duplicates rows; keyed on `(source_name, raw_file, raw_offset)`.
- **TF-IDF not SBERT**: lightweight, no GPU needed; the `embed_corpus` function in `fingerprint.py` is the exact swap point for a real sentence-transformer.
- **SQLite not Postgres**: same interface (`upsert`, `upsert_batch`, `all_records`); `storage.py` is the only module to change for a real DB migration.

---

## Layer 2 — Pattern Mining (`pattern_mining/`)

### Purpose
Clusters canonical problems into algorithmic patterns using semantic embeddings + HDBSCAN, then supports incremental assignment of new records without re-clustering.

### Pipeline Stages

| Stage | Module | Output |
|-------|--------|--------|
| Embedding generation | `embeddings.py` | 384-dim `all-MiniLM-L6-v2` vectors; cached at `data/processed/embeddings.npy` |
| Dimension reduction | `clustering.py` | UMAP: 384 → 64 dims (seeded, deterministic) |
| Clustering | `clustering.py` | HDBSCAN labels (−1 = noise/outlier) |
| Noise reassignment | `clustering.py` | Noise points with cosine sim ≥ 0.75 to nearest centroid are soft-assigned |
| Cluster summarization | `summarizer.py` | Auto-label (top tags + title n-grams), centroid, representative, difficulty dist |
| Persistence | `storage.py` | `patterns` + `pattern_assignments` tables in `ingestion.db` |
| Export | `pipeline.py` | `data/output/patterns.json` |

### Performance Metrics (last full run)

| Metric | Value |
|--------|-------|
| Records processed | **4,387** |
| Patterns discovered | **128** |
| Clustered records | **3,613** (82.4%) |
| Outlier records | **774** (17.6%) |
| Largest pattern | `pattern_5` — 208 problems (binary tree cluster) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| UMAP target dim | 64 |
| HDBSCAN min_cluster_size | 10, min_samples 1 |

### Incremental Assignment

New problems are assigned without re-clustering:
- Compute embedding → dot-product vs all stored centroid vectors
- If max cosine similarity ≥ 0.80: assign to that pattern, update centroid with running average
- Else: create new `pattern_{N}` row

---

## Layer 3 — Template Re-authoring (`transform/` + `sandbox/`)

### Purpose
Rewrites canonical problems into product-owned templates via an LLM, then validates every rewrite by running the **original canonical solver** against the template's sample tests — a template is only persisted if it's proven correct.

### Pipeline Stages

| Stage | Module | Notes |
|-------|--------|-------|
| Spec building | `spec_builder.py` | Deterministic INPUT_SPEC + few-shot exemplars |
| LLM call | `llm.py` | `MockLLMClient` (tests) or `OllamaLLMClient` (real) |
| JSON parse/repair | `reauthor.py` | Up to `max_attempts` re-prompts on malformed output |
| Validation | `validator.py` | Runs canonical solver via `sandbox/` subprocess harness |
| Storage | `storage.py` | `templates` + `variants` tables in `ingestion.db` |

### Hard Rejection Criteria (template never persisted if any are true)

- No Python canonical solution for the record
- No detectable entry point in the solution (`class Solution` method or top-level function)
- Zero `sample_public_tests` in LLM output
- Any sample test's claimed output doesn't match canonical solver's actual output
- Template argument count doesn't match canonical solver's parameter count

### Soft Review Flags (`needs_review=True` but still persisted)

- LLM didn't declare a `tie_breaker`
- Output schema shape differs from canonical record's
- Some fraction of randomized variants errored on the canonical solver

### Sandbox (`sandbox/runner.py`)

- Executes solution in a fresh `python -I` subprocess per test case
- Wall-clock timeout (5 s default) is the enforceable limit; `RLIMIT_AS` memory cap is best-effort (unreliable on macOS)
- Entry-point detection: `class Solution` first public method, or top-level public function
- Spot-check pass rate against real ingested data: **~70%** (documented gaps: custom structures like `ListNode`/`TreeNode`, multi-answer problems, in-place mutation contracts)

---

## Test Suite

| Scope | Files | Location |
|-------|-------|----------|
| Ingestion | 8 | `tests/` |
| Pattern mining | 5 | `pattern_mining/tests/` |
| Transform | 8 | `transform/tests/` |
| Sandbox | 1 | `sandbox/tests/` |
| **Total** | — | **135 passed, 0 failed** |

All transform tests use `MockLLMClient` (no network, fully deterministic).

---

## Output Artefacts

| File | Description |
|------|-------------|
| `data/output/ingestion.db` | Full SQLite store: records, patterns, assignments, templates |
| `data/output/unified_dataset.json` | Flat JSON array of 4,387 canonical records |
| `data/output/patterns.json` | 128 pattern summaries with members, tags, centroids |
| `data/output/patterns_dashboard.html` | Self-contained interactive HTML dashboard |
| `data/output/patterns_report.md` | Markdown table of all 128 patterns |
| `data/processed/embeddings.npy` | Cached 384-dim embeddings for 4,387 problems |
| `data/processed/ids.json` | Problem ID order matching `embeddings.npy` |

---

## Upgrade Paths

| Feature | Current Implementation | Upgrade Seam |
|---------|----------------------|--------------|
| NL Embeddings (ingestion) | TF-IDF (`fingerprint.embed_corpus`) | Replace `embed_corpus` with SBERT call |
| Code AST | Whitespace-normalized hash | Replace with tree-sitter per-language normalized AST |
| Pattern Embeddings | `all-MiniLM-L6-v2` (384-dim) | Change `DEFAULT_MODEL_NAME` in `embeddings.py` |
| LLM Client | `OllamaLLMClient` (Mistral-Instruct) | Add new `LLMClient` implementation to `llm.py` |
| Storage backend | SQLite | Reimplement `Storage` interface against Postgres/S3 |
| CI | None | Add GitHub Actions: `pytest` on PRs + nightly ingest smoke-test |

---

## CLI Quick Reference

```bash
# Ingestion
ingest \
  --source "markdown:data/raw/leetcode_doocs:doocs-leetcode:cc-by-sa-4.0:https://github.com/doocs/leetcode" \
  --source "json:data/raw/leetcode_problems.json:leetcode-problems:unknown" \
  --source "jsonl:data/raw/leetcode_dataset_train.jsonl:leetcode-dataset-train:educational" \
  --source "jsonl:data/raw/leetcode_dataset_test.jsonl:leetcode-dataset-test:educational" \
  --source "csv:data/raw/leetcode_kaggle.csv:leetcode-kaggle:unknown" \
  --db data/output/ingestion.db --output data/output/unified_dataset.json

# Pattern mining (full pipeline)
python -m pattern_mining.cli run --input data/output/unified_dataset.json

# Incremental assignment of new records
python -m pattern_mining.cli assign --input data/new_records.jsonl

# View patterns
python -m pattern_mining.cli view --export-html      # → data/output/patterns_dashboard.html
python -m pattern_mining.cli view --export-markdown  # → data/output/patterns_report.md
python -m pattern_mining.cli view --search "graph"

# Template re-authoring (mock client, smoke-test)
python -m transform.cli run --input data/output/unified_dataset.json --limit 5

# Template re-authoring (real Ollama model)
export TRANSFORM_LLM_MODE=ollama TRANSFORM_LLM_MODEL=mistral:instruct
python -m transform.cli run --limit 100

# Run full test suite
python -m pytest -q
```
