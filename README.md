# Dataset Ingestion Layer

Turns messy coding-problem source dumps (JSONL, CSV, HTML, and per-problem
Markdown repos like [doocs/leetcode](https://github.com/doocs/leetcode))
into a single clean, versioned, auditable dataset that downstream services
(pattern mining, template generation, validators, uniqueness checks) can
rely on.

`doocs/leetcode` is the primary/canonical source: it's a community-maintained,
actively-updated repo of ~4,400 per-problem Markdown pages (description,
examples, constraints, multi-language solutions), ingested first so it wins
dedup ties against the older JSON/JSONL/CSV snapshots, which now mainly
backfill metadata (acceptance rate, likes, alternate difficulty/tag labels)
for problems doocs doesn't fully cover.

This implementation follows the ingestion spec's architecture but swaps
heavyweight infra for local, dependency-light equivalents so the whole
thing runs standalone:

| Spec component        | This implementation |
|------------------------|----------------------|
| Postgres + JSONB       | SQLite with a JSON payload column (`storage.py`) |
| S3 raw blobs           | Local raw files, referenced by path in `ingest_trace` |
| FAISS / Pinecone / Elastic | In-process TF-IDF vectors + cosine similarity (`fingerprint.py`) |
| SBERT embeddings       | TF-IDF `Corpus` class — same interface (`embed_corpus`), swappable |
| tree-sitter AST        | Normalized-whitespace code hash (`code_ast_signature`) |
| Redis/RabbitMQ queue   | Not needed at this scale; pipeline runs in-process, batch-oriented |
| Kubernetes containers  | Not needed; single Python process |

Everything else — the canonical schema, the pipeline stages, the
dedup/validation/license policies — matches the spec directly.

## Architecture

```
Source files (JSONL/CSV/HTML/Markdown)
        |
        v
  Source Adapter        (adapters.py)   -> RawProblem (provenance attached)
        |
        v
  Normalizer             (normalizer.py) -> CanonicalRecord (clean text,
        |                                   canonical difficulty/tags,
        |                                   structured examples)
        v
  Schema Extractor        (schema_extractor.py) -> input/output JSON Schema
        |                                          + confidence score
        v
  Fingerprinting          (fingerprint.py) -> content_hash, nl_embedding,
        |                                     ast_signature
        v
  License/Provenance      (license_tracker.py) -> requires_legal_review,
        |                                          public_allowed flags
        v
  De-duplication          (dedup.py) -> consolidated canonical records,
        |                               duplicates marked + merged
        v
  Validation              (validation.py) -> needs_review flags + reasons
        |
        v
  Storage                 (storage.py) -> SQLite (idempotent upsert)
                                        -> unified_dataset.json export
```

`pipeline.py` (`IngestionPipeline`) wires all of this together and
returns `IngestMetrics` (records parsed/stored, duplicate rate,
needs-review rate, errors) per the observability section of the spec.

## Canonical schema

Every record converges on `CanonicalRecord` (`models.py`), matching the
spec's data model: `title`, `description`, `difficulty`, `tags`,
`input_schema`/`output_schema`, `examples`, `constraints`,
`canonical_solution`, `fingerprints`, `license_meta`, `ingest_trace`,
`version`/`history`, and `validation` flags.

`is_premium` / `data_unavailable`: `is_premium` marks problems doocs/leetcode
flags as LeetCode Premium-only (its `🔒` title suffix, stripped from
`title` and surfaced as this structured flag instead). Being premium does
**not** by itself mean data is missing — most premium problems still ship
a full description and a community solution. `data_unavailable` is the
narrower, more useful signal: it's only `True` when a problem is premium
**and** no source contributed a `canonical_solution` for it (as of the
last ingest, 29 problems). Dedup clears `data_unavailable` automatically
if a later-merged duplicate brings in a solution.

## De-duplication strategy

Implemented as four staged checks, cheapest first (`dedup.py`):

1. Exact match on normalized title.
2. Token-Jaccard overlap on title + first 50 description tokens.
3. Cosine similarity ≥ 0.90 on TF-IDF description embeddings.
4. Exact match on normalized code AST signature.

To avoid O(n²) comparisons, candidate pairs are pre-filtered by a title
token index (only records sharing at least one title token are compared).
Matches consolidate into the earliest-created record, which absorbs
`source_list`, `aliases`, and tags from the duplicate. Duplicates are kept
in storage (for audit) with `deduplicated_into` set, but excluded from
`unified_dataset.json` and `Storage.all_records()` by default.

**Sequel-problem guard**: LeetCode "Part II/III" sequel problems (e.g.
"Minimum Partition Score" vs "Minimum Partition Score II") deliberately
reuse their parent's setup text almost verbatim, which used to push them
past the token-Jaccard and embedding-similarity thresholds and get them
wrongly merged as duplicates. `dedup._is_sequel_pair` detects titles that
are identical except for a trailing roman-numeral/"Part N" suffix and
skips every near-duplicate stage for that pair, so sequels always survive
as distinct records.

## License & provenance policy

`license_tracker.classify_license` maps a declared license string to
`(requires_legal_review, public_allowed, internal_use_only)`:

- Known permissive licenses (MIT, Apache-2.0, BSD, CC0, CC-BY-4.0) →
  publishable, no review needed.
- Known internal-only licenses (`educational`, `research-only`,
  `proprietary`) → usable for internal pattern mining, never publishable,
  no review needed (the restriction is already known).
- `unknown` or any unrecognized license string → flagged
  `requires_legal_review`, not publishable, internal-use-only, until a
  human confirms otherwise.

Every record also carries `ingest_trace` (parser id, raw file path, raw
line/row offset, and — for the markdown adapter — the source repo's git
commit hash) so any field can be traced back to its exact source line.

`doocs/leetcode` is licensed CC-BY-SA-4.0 (share-alike, not in the
permissive allowlist above), so every doocs-sourced record is flagged
`requires_legal_review`/`internal_use_only` by design — treat the ingested
text as an internal reference corpus, not something to republish verbatim.

## Idempotency

Adapters emit a `raw_offset` (line number for JSONL, row number for CSV)
alongside the source file path. `Storage.upsert` looks up existing
records by `(source_name, raw_file, raw_offset)`; if a match exists, it
reuses that record's id and bumps `version`/`history` instead of
inserting a new row. Re-running the pipeline on an unchanged file is a
no-op in terms of record count (verified in
`tests/test_pipeline.py::test_pipeline_is_idempotent_on_rerun`).

## Usage

Set up the environment once:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Get the primary source (a one-time or periodic `git clone`/`pull` — the
pipeline itself only reads a local checkout, it does not fetch anything):

```bash
git clone --depth 1 https://github.com/doocs/leetcode.git data/raw/leetcode_doocs
```

Run an ingestion via the CLI. Each `--source` is
`kind:path:source_name[:license[:source_url]]`, where `kind` is one of
`jsonl`, `csv`, `html`, `markdown`. List `doocs-leetcode` first so it wins
dedup ties over the older auxiliary snapshots:

```bash
.venv/bin/ingest \
  --source "markdown:data/raw/leetcode_doocs:doocs-leetcode:cc-by-sa-4.0:https://github.com/doocs/leetcode" \
  --source "json:data/raw/leetcode_problems.json:leetcode-problems:unknown" \
  --source "jsonl:data/raw/leetcode_dataset_train.jsonl:leetcode-dataset-train:educational" \
  --source "jsonl:data/raw/leetcode_dataset_test.jsonl:leetcode-dataset-test:educational" \
  --source "csv:data/raw/leetcode_kaggle.csv:leetcode-kaggle:unknown" \
  --db data/output/ingestion.db \
  --output data/output/unified_dataset.json
```

This prints a JSON metrics summary (records parsed/stored, duplicate
rate, needs-review rate, errors) and writes:

- `data/output/ingestion.db` — SQLite store, all records (including
  deduplicated ones, for audit) with full JSON payload + version history.
- `data/output/unified_dataset.json` — flat array of surviving canonical
  records, ready for pattern mining / template generation.

## Extending this layer

- **Real embeddings**: replace `fingerprint.embed_corpus` with a call to
  a sentence-transformer model (or a hosted embeddings API). Everything
  downstream (dedup, similarity search) consumes plain vectors, so no
  other code changes.
- **Real AST analysis**: replace `fingerprint.code_ast_signature` with a
  tree-sitter-based normalized AST signature for cross-language,
  rename-invariant code duplicate detection.
- **New source types**: add an adapter class implementing
  `SourceAdapter.parse(path) -> Iterator[RawProblem]` and register it in
  `adapters.ADAPTER_REGISTRY`. `MarkdownAdapter` is one example: it walks
  a local checkout of a per-problem Markdown repo (doocs/leetcode's
  layout), recursively finding every `README.md`/`README_EN.md` that
  carries a `<!-- problem:start -->` marker (so category index pages are
  skipped), parsing YAML frontmatter for difficulty/tags, the first
  `## Description`/`题目描述` block for description/examples/constraints,
  and the first `## Solutions`/`解法` section's preferred-language code
  fence for a canonical solution.
- **Keeping doocs/leetcode fresh**: re-run `git pull` in
  `data/raw/leetcode_doocs` and re-run `ingest` with the same
  `--db`/`--output`. Idempotency is keyed on `(source_name, raw_file,
  raw_offset)` where `raw_file` is the problem's folder path — unchanged
  problems re-upsert into the same record id (bumping `version`), and any
  newly-added problem folders show up as new records automatically. There
  is no separate incremental/diff mode yet; each run reprocesses every
  file in the sources given, same as the other adapters.
- **Postgres/S3/vector DB**: `storage.py` is the only module that would
  need to change — the `Storage` interface (`upsert`, `upsert_batch`,
  `all_records`, cursors) can be reimplemented against Postgres/S3
  without touching the pipeline or adapters.

## Known limitations (by design, for this scope)

- Embeddings are TF-IDF, not semantic (SBERT-quality) — good enough to
  catch near-duplicate phrasing, not paraphrases.
- AST signatures are whitespace-normalized hashes, not real ASTs — they
  catch verbatim/near-verbatim code copies, not semantically-equivalent
  rewrites.
- Dedup candidate generation uses title-token blocking, not an ANN index,
  so pathologically large single-title clusters could get slow; fine for
  the batch sizes this is meant to run on.
- No queue/worker orchestration — the pipeline runs synchronously,
  in-process, per invocation.

## Solution code persistence

`canonical_solution.code` holds the actual source text for the first
language in `canonical_solution.languages` (set by whichever
adapter/normalizer step first captured a solution for that record).
Previously this text was hashed for AST/dedup purposes and then
discarded — never reaching `unified_dataset.json`/`ingestion.db` — which
made any kind of "run the canonical solution and check the output"
validation impossible downstream. It's now persisted end to end: adapter
→ `RawProblem.solution_code` → `normalizer.normalize` →
`CanonicalRecord.canonical_solution.code` → SQLite payload column →
`unified_dataset.json` export. As of the last ingest, 4,236 of 4,387
canonical records (96.6%) carry solution code.

Fixing this exposed a real, previously-latent bug in
`fingerprint.code_ast_signature`: its comment-stripping regex ran *after*
whitespace collapse, so on any solution starting with a multi-line
comment block (doocs's near-universal `# Definition for singly-linked
list...` header) it stripped from that `#` to the literal end of the
string, silently reducing the signature to `hash("")` for hundreds of
records and causing false-positive dedup merges between unrelated
problems that happened to share a comment-block prefix. Fixed to strip
comments per-line before collapsing whitespace (see
`tests/test_fingerprint.py::test_code_ast_signature_strips_comments_per_line_not_to_end_of_string`
for the regression test). Re-ingesting after the fix went from 255 bogus
`code_ast`-stage merges down to 9 legitimate ones (e.g. "Powx N" merging
with "Pow(x, n)" — same problem, different title across sources).

## Sandbox: running canonical solutions locally

`sandbox/` (new) executes a canonical Python solution against a
problem's examples and reports pass/fail per case — the foundation for
validating that an LLM-rewritten template preserves the original
algorithmic behavior. Explicitly **not** production-grade isolation: it
runs code in a fresh `python -I` subprocess with a wall-clock timeout and
a best-effort `RLIMIT_AS` memory cap (unreliable on macOS — confirmed
during development; the timeout is the limit you can actually rely on
everywhere). Only run code whose provenance you already trust.

```python
from sandbox import run_solution_on_examples

result = run_solution_on_examples(canonical_record_dict)
result.outcome        # RunOutcome.OK / NO_ENTRY_POINT / UNSUPPORTED_LANGUAGE / NO_EXAMPLES
result.all_passed     # True only if outcome is OK and every case passed
result.cases          # per-example CaseResult: actual, expected_parsed, passed, error, timed_out
```

Entry-point detection looks for a `class Solution`'s first public method,
or failing that, a top-level public function — the two shapes every
doocs/leetcode Python snippet uses. A spot-check against 40 random
problems from the real ingested dataset passes ~70% end to end; documented
gaps (not bugs) account for the rest: problems requiring custom
structures (`ListNode`/`TreeNode`), LeetCode's "if multiple answers
exist, return any" problems (exact-match comparison gives false
negatives), and in-place-mutation return contracts (LeetCode's
`removeElement`-style "return k, and the first k elements of nums..."
problems, which this harness doesn't attempt to check).

## Template re-authoring pipeline (transform/)

`transform/` turns a canonical, ingested problem into a rewritten,
product-owned "template" via an LLM, then validates the rewrite by
running the **canonical solver** (never the LLM's own output) against
it, so a rewrite is only ever persisted if it's proven to preserve the
original algorithmic behavior.

```
canonical record (unified_dataset.json)
        |
        v
  spec_builder.build_spec()      -> deterministic INPUT_SPEC + seed
        |
        v
  spec_builder.render_prompt()   -> SYSTEM + USER prompt (few-shot exemplars baked in)
        |
        v
  llm.LLMClient.call()           -> raw model response text
        |
        v
  reauthor.parse_template_output()  -> TemplateOutput (strict JSON schema)
        |    (malformed? re-prompt up to max_attempts, then reject)
        v
  validator.validate_template()
        |    - runs canonical solver against the template's own sample
        |      tests, with args positionally remapped onto the solver's
        |      real parameter names (a rewrite is free to rename fields,
        |      not to add/remove/reorder them)
        |    - generates N deterministic randomized variants (gated by
        |      the LLM's own declared value/n ranges) and re-runs the
        |      canonical solver on each as an additional parity signal
        v
  storage.write_template() / write_variants() / link_pattern_to_template()
        -> `templates` + `variants` tables in ingestion.db
```

### Data model (`transform/models.py`)

- `TemplateOutput` — the strict JSON shape the LLM must return (title,
  description, input/output schema, constraints, tie-breaker, sample
  tests, variant-generation hints). Parsed with `extra="ignore"` so an
  LLM tacking on an unexpected field doesn't break parsing, but every
  *required* field missing still fails validation and triggers a
  re-prompt.
- `Template` / `Variant` — the persisted, validated records. `Template`
  carries a `Provenance` block (`canonical_problem_id`, `source_list`,
  `doocs_commit_hash`, `license`, `transform_model`, `transform_seed`)
  and `needs_review`/`review_reasons` for soft (non-rejecting) concerns.

### DB schema (`transform/storage.py`)

New tables in the same `ingestion.db` SQLite file pattern_mining already
uses (`templates`, `variants`), plus a `template_id` column added to
pattern_mining's existing `patterns` table (idempotent `ALTER TABLE`,
skipped harmlessly if `patterns` doesn't exist yet). Upserts follow the
same `INSERT ... ON CONFLICT DO UPDATE` style as
`pattern_mining/storage.py`.

### Validation & rejection criteria (`transform/validator.py`)

A template is **hard-rejected** (never persisted) if:
- no Python canonical solution exists for that record at all,
- the canonical solver has no detectable entry point,
- the template declared zero `sample_public_tests`,
- any sample test's claimed output isn't reproduced when the canonical
  solver is actually run on it (the core safety gate),
- the template's sample-test argument count doesn't match the canonical
  solver's real parameter count (a strong signal the rewrite added or
  dropped a required field).

A template is **accepted but flagged `needs_review`** if: it passed
every sample test, but the LLM didn't declare a `tie_breaker`, its
declared `output_schema` shape differs from the canonical record's
(e.g. array vs scalar), or some fraction of the randomized variants
errored on the canonical solver (may mean the LLM's declared ranges need
tightening, not that the rewrite itself is unfaithful).

Randomized variant generation is a simple deterministic scalar/array
randomizer gated by the LLM's own `notes_for_variant_generator` ranges —
not a full per-pattern `variant_engine/` (doesn't exist yet). Problems
whose schema needs custom structures or nested arrays-of-arrays skip
variant generation with a stated reason rather than guessing.

### LLM wrapper (`transform/llm.py`)

`MockLLMClient` (deterministic, no network, used by every current test)
and `OllamaLLMClient` (real HTTP path via stdlib `urllib` — **not yet
verified against a running model**, since no local model runtime was
available in this environment). `get_default_client()` picks between
them via `TRANSFORM_LLM_MODE` (`mock` default, or `ollama`). See the
module docstring for how to install Ollama and pull `mistral:instruct`
(or the heavier `llama2:13b` fallback) to exercise the real path.

### Usage

```bash
# Smoke-test on 5 records with whatever TRANSFORM_LLM_MODE is set to
# (defaults to the mock client, which will reject everything cleanly —
# useful for confirming the pipeline wiring without a real model):
.venv/bin/python -m transform.cli run \
  --input data/output/unified_dataset.json \
  --db data/output/ingestion.db \
  --limit 5

# Once a real model is configured (see transform/llm.py docstring):
export TRANSFORM_LLM_MODE=ollama
export TRANSFORM_LLM_MODEL=mistral:instruct
.venv/bin/python -m transform.cli run --limit 100
```

Prints a JSON metrics summary (`records_considered`, `accepted`,
`rejected_invalid_json`, `rejected_validation_failed`,
`acceptance_rate`, `needs_review`, `llm_errors`,
`total_variants_generated`) and, to stderr, up to 10 rejection reasons
for triage. Incremental by default — a canonical record already holding
a template is skipped (no LLM call) unless `--force` is passed.

### Not built yet

- A real `variant_engine/` (structural, per-pattern variant generation
  beyond scalar/array randomization).
- `data/final_bank/` export (the design doc's versioned publish step).
- CI wiring (a GitHub Action running `pytest transform/tests` on PRs, a
  small-scale smoke-test job) — not applicable in this environment since
  there's no `.git`/CI system present, but straightforward to add once
  there is one.
- Verification against a real local model — everything here is proven
  correct against the mock client and, for the validator/storage/sandbox
  layers, against real canonical records pulled from the actual ingested
  dataset. The LLM call itself is the one piece that needs a live model
  to fully confirm.
