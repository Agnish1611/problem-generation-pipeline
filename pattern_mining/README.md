# Pattern Mining Module

The `pattern_mining` module clusters canonical coding problems into distinct algorithmic patterns, summarizes discovered clusters, persists them into SQLite, and supports incremental assignment for newly ingested problems without requiring a full re-clustering run.

---

## Architecture Overview

```
problem-generation-pipeline/
├── ingestion/                 # Source adapters, normalization, and deduplication
├── pattern_mining/
│   ├── __init__.py
│   ├── embeddings.py          # Embedding generation & atomic disk caching
│   ├── clustering.py          # UMAP dimension reduction & HDBSCAN clustering
│   ├── summarizer.py          # Cluster summarization & representative selection
│   ├── assigner.py            # Incremental assignment & running centroid updates
│   ├── storage.py             # SQLite migration, pattern/assignment persistence
│   ├── pipeline.py            # Full end-to-end pipeline orchestrator
│   ├── cli.py                 # CLI interface (run, assign, view subcommands)
│   └── tests/
│       ├── test_embeddings.py
│       ├── test_clustering.py
│       ├── test_assigner.py
│       ├── test_storage.py
│       └── test_viewer.py
```

---

## Quickstart & CLI Usage

Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 1. Full Pipeline Run

Ingests `unified_dataset.json`, computes embeddings, runs UMAP+HDBSCAN, summarizes patterns, updates `ingestion.db`, and outputs `patterns.json`:

```bash
python -m pattern_mining.cli run \
  --input data/output/unified_dataset.json \
  --db data/output/ingestion.db \
  --output data/output/patterns.json
```

Optional tuning parameters:
* `--min-cluster-size <int>`: Minimum cluster size for HDBSCAN (default: `10`).
* `--min-samples <int>`: Conservative clustering parameter (default: `5`).
* `--umap-dim <int>`: Dimension to reduce embeddings to before clustering (default: `64`).
* `--recompute-embeddings`: Force re-encoding of problem texts even if cache exists.
* `--cache-dir <path>`: Directory for cached `.npy` and `.json` artifacts (default: `data/processed`).

### 2. Incremental Assignment

Assigns new problem records from a JSONL or JSON file to existing patterns without re-clustering:

```bash
python -m pattern_mining.cli assign \
  --input data/new_records.jsonl \
  --db data/output/ingestion.db \
  --threshold 0.80
```

Optional parameters:
* `--threshold <float>`: Cosine similarity threshold for assignment (default: `0.80`).
* `--batch-size <int>`: Batch size for sentence-transformer embedding (default: `64`).

### 3. Human-Readable Viewing & Exploration

#### Terminal Table
View patterns sorted by size with representatives, tags, and difficulty breakdown:
```bash
# View top 20 patterns
python -m pattern_mining.cli view --limit 20

# Search for patterns related to "binary-tree" or "graph"
python -m pattern_mining.cli view --search "graph"

# Inspect a specific pattern and list all its member problems
python -m pattern_mining.cli view --pattern pattern_0
```

#### Interactive Web Dashboard
Open the standalone interactive HTML dashboard in your browser:
```bash
open data/output/patterns_dashboard.html
```
Or regenerate it anytime:
```bash
python -m pattern_mining.cli view --export-html
```

#### Markdown Report
View the generated summary table in markdown:
* File: `data/output/patterns_report.md`

---

## Configuration & Constants

| Parameter | Default | Component | Description |
|-----------|---------|-----------|-------------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` | `embeddings.py` | 384-dimensional dense semantic embedding model. Fast on CPU. |
| UMAP Dimension | `64` | `clustering.py` | Intermediate dimension for UMAP before HDBSCAN. |
| UMAP Random State | `42` | `clustering.py` | Ensures deterministic UMAP dimensionality reduction. |
| Min Cluster Size | `10` | `clustering.py` | Minimum points to form an HDBSCAN cluster. |
| Min Samples | `1` | `clustering.py` | Neighborhood density threshold in HDBSCAN. |
| Selection Method | `eom` | `clustering.py` | Excess of Mass cluster selection in HDBSCAN. |
| Assignment Threshold | `0.80` | `assigner.py` | Minimum cosine similarity to assign to an existing pattern. |

---

## Incremental Update Mechanism

When new records are ingested:
1. **Deduplication / Skip**: The assigner inspects `pattern_assignments` and skips any problem whose `id` or `content_hash` has already been assigned.
2. **Embedding & Similarity**: Dense embeddings are generated and compared via vectorized matrix dot-product against all stored pattern centroids.
3. **Assignment or Creation**:
   * If $\max(\text{similarity}) \ge 0.80$: The record is assigned to that pattern. The pattern's centroid is updated using a streaming running average formula:
     $$\mathbf{c}_{\text{new}} = \frac{\mathbf{c}_{\text{old}} \cdot n_{\text{old}} + \mathbf{e}_{\text{new}}}{n_{\text{old}} + 1}$$
     The pattern's `size` is incremented, and changes are written to the database.
   * If $\max(\text{similarity}) < 0.80$: A new pattern row (`pattern_{next_int}`) is created with the new problem as its representative, initialized with `size = 1` and `centroid = e_new`.

---

## Database Schema

Tables created in `data/output/ingestion.db`:

### `patterns`
* `pattern_id` (TEXT PRIMARY KEY): Unique identifier (e.g. `pattern_0`).
* `label` (TEXT): Auto-generated algorithmic label using top tags and title n-grams.
* `size` (INTEGER): Number of problems currently assigned to this pattern.
* `representative` (TEXT): Problem ID closest to the centroid by cosine distance.
* `top_tags` (TEXT): JSON array of the top tags by frequency.
* `difficulty_json` (TEXT): JSON dictionary of difficulty distribution.
* `centroid` (BLOB): Float32 binary bytes representing the pattern centroid embedding.
* `created_at` (TEXT): ISO 8601 timestamp.

### `pattern_assignments`
* `problem_id` (TEXT PRIMARY KEY): Canonical problem ID.
* `pattern_id` (TEXT): Assigned pattern ID.
* `confidence` (REAL): Assignment confidence (1.0 for initial members, cosine similarity for incremental).
* `assigned_at` (TEXT): ISO 8601 timestamp.

---

## Running Tests

Run unit tests across the entire test suite:
```bash
pytest pattern_mining/tests/ -v
```
Or run all repository tests:
```bash
pytest tests/ pattern_mining/tests/ -v
```
