"""Source Adapters.

One adapter per source type. Each adapter's job is narrow: read a raw file
and yield `RawProblem` instances with provenance attached. No cleaning or
canonicalization happens here — that's the Normalizer's job.

Malformed records are skipped and logged (spec section 12) rather than
raising, so one bad line does not kill an entire ingest run.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Iterator

from bs4 import BeautifulSoup

from .models import RawProblem

logger = logging.getLogger(__name__)


class AdapterError(Exception):
    """Raised for adapter-level failures that should stop the whole run
    (e.g. file not found). Per-record parse failures are logged and
    skipped instead of raised."""


class SourceAdapter:
    """Base interface for all source adapters."""

    parser_id: str = "base_adapter_v1"

    def __init__(self, source_name: str, license: str = "unknown", source_url: str | None = None):
        self.source_name = source_name
        self.license = license
        self.source_url = source_url

    def parse(self, path: str | Path) -> Iterator[RawProblem]:
        raise NotImplementedError

    def _base_kwargs(self, raw_file: str, raw_offset: int | None) -> dict:
        return dict(
            source_name=self.source_name,
            source_url=self.source_url,
            license=self.license,
            raw_file=raw_file,
            raw_offset=raw_offset,
            parser=self.parser_id,
        )


class JSONLAdapter(SourceAdapter):
    """Parses newline-delimited JSON dumps (e.g. LeetCodeDataset-style)."""

    parser_id = "parse_jsonl_v1"

    # Field name candidates, since dumps disagree on naming.
    FIELD_MAP = {
        "title": ("title", "problem_title", "name"),
        "description": ("description", "content", "problem_description", "body"),
        "difficulty": ("difficulty", "level"),
        "tags": ("tags", "topics", "categories"),
        "examples": ("examples", "test_cases", "sample_tests", "input_output"),
        "constraints": ("constraints",),
        "solution_code": ("solution", "code", "canonical_solution", "completion"),
        "solution_language": ("language", "lang"),
    }

    def _get(self, obj: dict, key: str):
        for candidate in self.FIELD_MAP[key]:
            if candidate in obj and obj[candidate] not in (None, ""):
                return obj[candidate]
        return None

    def parse(self, path: str | Path) -> Iterator[RawProblem]:
        path = Path(path)
        if not path.exists():
            raise AdapterError(f"JSONL source not found: {path}")

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for offset, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSON at %s:%d (%s)", path, offset, exc)
                    continue

                title = self._get(obj, "title")
                if not title:
                    slug = obj.get("task_id") or obj.get("problem_slug")
                    if slug:
                        title = str(slug).replace("-", " ").title()
                if not title:
                    logger.warning("Skipping record without title at %s:%d", path, offset)
                    continue

                examples_raw = self._get(obj, "examples") or []
                if isinstance(examples_raw, dict):
                    examples_raw = [examples_raw]

                constraints = self._get(obj, "constraints") or []
                if isinstance(constraints, str):
                    constraints = [constraints]

                tags = self._get(obj, "tags") or []
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]

                solution_code = self._get(obj, "solution_code")
                solution_language = self._get(obj, "solution_language")
                if not solution_language and solution_code:
                    solution_language = "python"

                yield RawProblem(
                    title=str(title),
                    description=str(self._get(obj, "description") or ""),
                    difficulty=self._get(obj, "difficulty"),
                    tags=list(tags),
                    examples_raw=examples_raw,
                    constraints=list(constraints),
                    solution_code=solution_code,
                    solution_language=solution_language,
                    extra=obj,
                    **self._base_kwargs(str(path), offset),
                )


class JSONAdapter(SourceAdapter):
    """Parses standard JSON files containing either a list of problems or
    a top-level dict containing a 'questions' or 'problems' list (e.g.
    leetcode_problems.json)."""

    parser_id = "parse_json_v1"
    FIELD_MAP = JSONLAdapter.FIELD_MAP

    def _get(self, obj: dict, key: str):
        for candidate in self.FIELD_MAP[key]:
            if candidate in obj and obj[candidate] not in (None, ""):
                return obj[candidate]
        return None

    def _extract_examples(self, raw_examples: list) -> list[dict]:
        results = []
        for ex in raw_examples:
            if not isinstance(ex, dict):
                continue
            if "input" in ex and "output" in ex:
                results.append(ex)
                continue
            txt = ex.get("example_text") or ""
            if txt:
                inp_match = re.search(r"Input:\s*(.*?)(?=\s*Output:|$)", txt, re.DOTALL | re.IGNORECASE)
                out_match = re.search(r"Output:\s*(.*?)(?=\s*Explanation:|$)", txt, re.DOTALL | re.IGNORECASE)
                exp_match = re.search(r"Explanation:\s*(.*?)$", txt, re.DOTALL | re.IGNORECASE)
                inp = inp_match.group(1).strip() if inp_match else ""
                out = out_match.group(1).strip() if out_match else ""
                exp = exp_match.group(1).strip() if exp_match else None
                if inp or out:
                    item = {"input": inp, "output": out}
                    if exp:
                        item["explanation"] = exp
                    results.append(item)
        return results

    def parse(self, path: str | Path) -> Iterator[RawProblem]:
        path = Path(path)
        if not path.exists():
            raise AdapterError(f"JSON source not found: {path}")

        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise AdapterError(f"Failed to read/parse JSON source {path}: {exc}") from exc

        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("questions") or data.get("problems") or data.get("data") or [data]
        else:
            raise AdapterError(f"Unexpected root type in JSON {path}: {type(data)}")

        for offset, obj in enumerate(items):
            if not isinstance(obj, dict):
                continue

            title = self._get(obj, "title")
            if not title:
                slug = obj.get("task_id") or obj.get("problem_slug")
                if slug:
                    title = str(slug).replace("-", " ").title()
            if not title:
                logger.warning("Skipping JSON record without title at %s:%d", path, offset)
                continue

            raw_examples = self._get(obj, "examples") or []
            if isinstance(raw_examples, dict):
                raw_examples = [raw_examples]
            examples_raw = self._extract_examples(raw_examples)

            constraints = self._get(obj, "constraints") or []
            if isinstance(constraints, str):
                constraints = [constraints]

            tags = self._get(obj, "tags") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            solution_code = self._get(obj, "solution_code")
            solution_language = self._get(obj, "solution_language")
            if not solution_code and isinstance(obj.get("code_snippets"), dict):
                snippets = obj["code_snippets"]
                if "python3" in snippets:
                    solution_code = snippets["python3"]
                    solution_language = "python"
                elif "python" in snippets:
                    solution_code = snippets["python"]
                    solution_language = "python"
                elif snippets:
                    first_lang = next(iter(snippets))
                    solution_code = snippets[first_lang]
                    solution_language = first_lang

            yield RawProblem(
                title=str(title),
                description=str(self._get(obj, "description") or ""),
                difficulty=self._get(obj, "difficulty"),
                tags=list(tags),
                examples_raw=examples_raw,
                constraints=list(constraints),
                solution_code=solution_code,
                solution_language=solution_language,
                extra=obj,
                **self._base_kwargs(str(path), offset),
            )


class CSVAdapter(SourceAdapter):
    """Parses CSV/Kaggle-style snapshots. Column headers are matched
    case-insensitively against the same field candidates as JSONLAdapter."""

    parser_id = "parse_csv_v1"

    FIELD_MAP = JSONLAdapter.FIELD_MAP

    def _get(self, row: dict, key: str):
        lower_row = {k.lower(): v for k, v in row.items()}
        for candidate in self.FIELD_MAP[key]:
            val = lower_row.get(candidate.lower())
            if val:
                return val
        return None

    def parse(self, path: str | Path) -> Iterator[RawProblem]:
        path = Path(path)
        if not path.exists():
            raise AdapterError(f"CSV source not found: {path}")

        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for offset, row in enumerate(reader):
                title = self._get(row, "title")
                if not title:
                    logger.warning("Skipping CSV row without title at %s:%d", path, offset)
                    continue

                tags_raw = self._get(row, "tags") or ""
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else []

                constraints_raw = self._get(row, "constraints") or ""
                constraints = [c.strip() for c in constraints_raw.split(";") if c.strip()] if isinstance(constraints_raw, str) else []

                yield RawProblem(
                    title=str(title),
                    description=str(self._get(row, "description") or ""),
                    difficulty=self._get(row, "difficulty"),
                    tags=tags,
                    examples_raw=[],
                    constraints=constraints,
                    solution_code=self._get(row, "solution_code"),
                    solution_language=self._get(row, "solution_language"),
                    extra=dict(row),
                    **self._base_kwargs(str(path), offset),
                )


class HTMLAdapter(SourceAdapter):
    """Parses a directory of scraped HTML problem pages.

    Heuristic extraction: <h1>/<title> for the title, the largest text
    block for the description, and any element tagged with a `data-tag`
    or class containing "tag" for tags. This is intentionally simple —
    real scraped corpora vary wildly in structure, so treat this as a
    starting point that will need per-source tuning.
    """

    parser_id = "parse_html_v1"

    def parse(self, path: str | Path) -> Iterator[RawProblem]:
        path = Path(path)
        if not path.exists():
            raise AdapterError(f"HTML source not found: {path}")

        files = [path] if path.is_file() else sorted(path.glob("*.html"))
        for offset, file in enumerate(files):
            try:
                html = file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Skipping unreadable HTML file %s (%s)", file, exc)
                continue

            soup = BeautifulSoup(html, "html.parser")

            title_el = soup.find("h1") or soup.find("title")
            title = title_el.get_text(strip=True) if title_el else None
            if not title:
                logger.warning("Skipping HTML file without title at %s", file)
                continue

            desc_el = soup.find("div", class_=lambda c: bool(c and "description" in c)) or soup.find("article") or soup.find("body")
            description = desc_el.get_text("\n", strip=True) if desc_el else ""

            tag_els = soup.find_all(class_=lambda c: bool(c and "tag" in c))
            tags = [t.get_text(strip=True) for t in tag_els if t.get_text(strip=True)]

            yield RawProblem(
                title=title,
                description=description,
                difficulty=None,
                tags=tags,
                examples_raw=[],
                constraints=[],
                extra={},
                **self._base_kwargs(str(file), offset),
            )


class MarkdownAdapter(SourceAdapter):
    """Parses a directory of per-problem markdown files laid out like the
    doocs/leetcode GitHub repo: one folder per problem containing a
    `README_EN.md` (and/or `README.md`) with YAML frontmatter
    (`difficulty`, `tags`) followed by an `H1` title line, a
    `## Description` section wrapped in `<!-- description:start/end -->`
    markers, and language-tagged fenced code blocks under `## Solutions`.

    Folders that only ship a Chinese README (no English version — true
    for the `lcof`/`lcof2`/`lcp`/`lcs` problem sets in that repo) fall
    back to `README.md` so no problems are silently skipped.

    Each problem folder is treated as one record; `raw_offset` is a
    stable index (folder's sorted position within the walk) rather than
    a byte offset, since idempotent upserts key on
    `(source_name, raw_file, raw_offset)` and `raw_file` (the folder
    path) is already unique per problem here.
    """

    parser_id = "parse_markdown_v1"

    # Matches numeric-id / "no. XX." style prefixes doocs/leetcode uses in
    # its H1 title line, so the canonical `title` field doesn't end up
    # cluttered with problem numbers (e.g. "1. Two Sum" -> "Two Sum").
    _TITLE_PREFIX_RE = re.compile(
        r"^(?:LCP|LCS)\s*\d+\.\s*"
        r"|^面试题\s*\d+\s*-?\s*[IVXivx]*\.\s*"
        r"|^剑指\s*Offer(?:\s*II)?\s*\d+\.\s*"
        r"|^\d+(?:\.\d+)*\.\s*",
    )

    _FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
    _H1_RE = re.compile(r"^#\s+\[(.*?)\]\((.*?)\)\s*$", re.MULTILINE)
    # doocs/leetcode suffixes the H1 title with "🔒" for LeetCode
    # Premium-only problems (e.g. "631. Design Excel Sum Formula 🔒").
    _PREMIUM_MARKER_RE = re.compile(r"\s*🔒\s*$")
    _DESCRIPTION_RE = re.compile(
        r"<!--\s*description:start\s*-->(.*?)<!--\s*description:end\s*-->", re.DOTALL
    )
    # doocs/leetcode has used two HTML shapes for examples over the years:
    # older problems wrap them in a bare <pre>...</pre>, newer ones use
    # <div class="example-block">...</div>. Capture either as one block of
    # raw HTML, then pull Input/Output/Explanation out of that block.
    _EXAMPLE_CONTAINER_RE = re.compile(
        r"<pre>(.*?)</pre>|<div class=\"example-block\">(.*?)</div>", re.DOTALL
    )
    # doocs/leetcode's zh-only problem sets (lcof/lcof2/lcp/lcs) ship no
    # README_EN.md at all, so example/constraint labels must also be
    # matched in Chinese, not just English.
    _EXAMPLE_INPUT_RE = re.compile(
        r"<strong>(?:Input|输入)[：:]?</strong>\s*(.*?)(?=<strong>(?:Output|输出)[：:]?</strong>|$)", re.DOTALL
    )
    _EXAMPLE_OUTPUT_RE = re.compile(
        r"<strong>(?:Output|输出)[：:]?</strong>\s*(.*?)(?=<strong>(?:Explanation|解释)[：:]?</strong>|$)", re.DOTALL
    )
    _EXAMPLE_EXPLANATION_RE = re.compile(r"<strong>(?:Explanation|解释)[：:]?</strong>\s*(.*?)$", re.DOTALL)
    _EXAMPLE_HAS_INPUT_RE = re.compile(r"<strong>(?:Input|输入)[：:]?</strong>")
    _CONSTRAINTS_RE = re.compile(
        r"<strong>(?:Constraints|限制|提示)[：:]?</strong>.*?<ul>(.*?)</ul>", re.DOTALL
    )
    _LI_RE = re.compile(r"<li>(.*?)</li>", re.DOTALL)
    _CODE_FENCE_RE = re.compile(r"^####\s+(\S+)\s*\n+```[a-zA-Z0-9]*\n(.*?)```", re.DOTALL | re.MULTILINE)

    # doocs/leetcode's per-language headings -> our canonical language ids.
    _LANGUAGE_HEADING_MAP = {
        "python3": "python",
        "python": "python",
        "java": "java",
        "c++": "cpp",
        "go": "go",
        "typescript": "typescript",
        "javascript": "javascript",
        "rust": "rust",
        "c#": "csharp",
    }
    _PREFERRED_LANGUAGES = ("python3", "python", "java", "c++")

    def __init__(self, source_name: str, license: str = "unknown", source_url: str | None = None, commit_hash: str | None = None):
        super().__init__(source_name=source_name, license=license, source_url=source_url)
        self.commit_hash = commit_hash

    def _strip_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        # Preserve exponent notation (e.g. "10<sup>4</sup>" -> "10^4") before
        # flattening, otherwise BeautifulSoup's get_text would either glue
        # the base and exponent together ("104") or split them onto
        # separate lines depending on the chosen separator.
        for sup in soup.find_all("sup"):
            sup.insert_before("^")
        # Block-level tags (p, li, div, pre) get a newline separator so
        # paragraphs/list items don't run together; a plain get_text("\n")
        # would also break every inline tag (code, strong, em, sup) onto
        # its own line, which is why sup is handled separately above.
        for tag in soup.find_all(["p", "li", "div", "pre", "br"]):
            tag.append("\n")
        return soup.get_text("").strip()

    def _extract_title(self, text: str) -> tuple[str, str | None, bool]:
        m = self._H1_RE.search(text)
        if not m:
            return "", None, False
        raw_title, url = m.group(1).strip(), m.group(2).strip()
        is_premium = bool(self._PREMIUM_MARKER_RE.search(raw_title))
        raw_title = self._PREMIUM_MARKER_RE.sub("", raw_title).strip()
        title = self._TITLE_PREFIX_RE.sub("", raw_title, count=1).strip()
        return title or raw_title, url, is_premium

    def _extract_description(self, text: str) -> str:
        m = self._DESCRIPTION_RE.search(text)
        if not m:
            return ""
        return self._strip_html(m.group(1)).strip()

    def _extract_examples(self, description_html: str) -> list[dict]:
        examples = []
        for container_match in self._EXAMPLE_CONTAINER_RE.finditer(description_html):
            block = container_match.group(1) or container_match.group(2) or ""
            if not self._EXAMPLE_HAS_INPUT_RE.search(block):
                continue

            inp_match = self._EXAMPLE_INPUT_RE.search(block)
            out_match = self._EXAMPLE_OUTPUT_RE.search(block)
            exp_match = self._EXAMPLE_EXPLANATION_RE.search(block)

            inp = self._strip_html(inp_match.group(1)).strip() if inp_match else ""
            out = self._strip_html(out_match.group(1)).strip() if out_match else ""
            explanation = self._strip_html(exp_match.group(1)).strip() if exp_match else None

            if inp or out:
                item = {"input": inp, "output": out}
                if explanation:
                    item["explanation"] = explanation
                examples.append(item)
        return examples

    def _extract_constraints(self, description_html: str) -> list[str]:
        m = self._CONSTRAINTS_RE.search(description_html)
        if not m:
            return []
        return [self._strip_html(li).strip() for li in self._LI_RE.findall(m.group(1)) if li.strip()]

    def _extract_solution(self, text: str) -> tuple[str | None, str | None]:
        """Pulls the first fenced code block for a preferred language out
        of the first `## Solutions` section (Solution 1 / 方法一), so we
        don't end up mixing in later alternative solutions.
        """
        solution_section = text.split("## Solutions")
        if len(solution_section) < 2:
            solution_section = text.split("## 解法")
        if len(solution_section) < 2:
            return None, None
        body = solution_section[1]
        # Stop at the end of the first `<!-- solution:end -->` block so we
        # only look at "Solution 1", not every alternative approach.
        end_idx = body.find("<!-- solution:end -->")
        if end_idx != -1:
            body = body[:end_idx]

        blocks = {heading.lower(): code for heading, code in self._CODE_FENCE_RE.findall(body)}
        for lang in self._PREFERRED_LANGUAGES:
            if lang in blocks:
                return blocks[lang].strip(), self._LANGUAGE_HEADING_MAP.get(lang, lang)
        if blocks:
            heading, code = next(iter(blocks.items()))
            return code.strip(), self._LANGUAGE_HEADING_MAP.get(heading, heading)
        return None, None

    def _iter_problem_dirs(self, root: Path) -> Iterable[Path]:
        """doocs/leetcode nests problem folders at different depths per
        problem set: three levels under `solution/<range>/<id>.<title>/`
        but only one level under `lcof/`, `lcof2/`, `lcp/`, `lcs/`,
        `lcci/`. Rather than assume a fixed depth (which would silently
        miss whole subtrees), recursively find every `README.md`/
        `README_EN.md` and keep only the ones that actually look like a
        single-problem page (carries the `<!-- problem:start -->`
        marker) rather than a category index page like `solution/README.md`.
        """
        candidates: dict[Path, None] = {}
        for pattern in ("**/README.md", "**/README_EN.md"):
            for readme in root.glob(pattern):
                if ".git" in readme.parts:
                    continue
                parent = readme.parent
                if parent in candidates:
                    continue
                try:
                    head = readme.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if "<!-- problem:start -->" in head:
                    candidates[parent] = None
        return sorted(candidates)

    def _detect_commit_hash(self, root: Path) -> str | None:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        return None

    def parse(self, path: str | Path) -> Iterator[RawProblem]:
        root = Path(path)
        if not root.exists():
            raise AdapterError(f"Markdown source directory not found: {root}")

        commit_hash = self.commit_hash or self._detect_commit_hash(root)
        problem_dirs = self._iter_problem_dirs(root)
        for offset, problem_dir in enumerate(problem_dirs):
            readme = problem_dir / "README_EN.md"
            if not readme.exists():
                readme = problem_dir / "README.md"
            if not readme.exists():
                continue

            try:
                text = readme.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Skipping unreadable markdown file %s (%s)", readme, exc)
                continue

            fm_match = self._FRONTMATTER_RE.match(text)
            frontmatter: dict = {}
            if fm_match:
                try:
                    import yaml

                    frontmatter = yaml.safe_load(fm_match.group(1)) or {}
                except Exception:  # noqa: BLE001 - tolerate malformed frontmatter
                    frontmatter = {}

            title, problem_url, is_premium = self._extract_title(text)
            if not title:
                logger.warning("Skipping markdown problem without a title at %s", readme)
                continue

            description_html_match = self._DESCRIPTION_RE.search(text)
            description_html = description_html_match.group(1) if description_html_match else ""
            description = self._extract_description(text)
            examples_raw = self._extract_examples(description_html)
            constraints = self._extract_constraints(description_html)
            solution_code, solution_language = self._extract_solution(text)

            tags = frontmatter.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]

            yield RawProblem(
                title=title,
                description=description,
                difficulty=frontmatter.get("difficulty"),
                tags=list(tags),
                examples_raw=examples_raw,
                constraints=constraints,
                solution_code=solution_code,
                solution_language=solution_language,
                is_premium=is_premium,
                extra={"problem_url": problem_url, "folder": str(problem_dir)},
                **self._base_kwargs(str(problem_dir), offset),
                commit_hash=commit_hash,
            )


ADAPTER_REGISTRY: dict[str, type[SourceAdapter]] = {
    "jsonl": JSONLAdapter,
    "json": JSONAdapter,
    "csv": CSVAdapter,
    "html": HTMLAdapter,
    "markdown": MarkdownAdapter,
}


def get_adapter(kind: str, **kwargs) -> SourceAdapter:
    try:
        cls = ADAPTER_REGISTRY[kind]
    except KeyError as exc:
        raise AdapterError(f"Unknown adapter kind: {kind!r}. Available: {list(ADAPTER_REGISTRY)}") from exc
    return cls(**kwargs)
