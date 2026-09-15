"""Minimal, dev-only sandbox for running canonical Python solutions.

SCOPE & LIMITATIONS (read before relying on this for anything sensitive):

  * This is process isolation, NOT a security sandbox. It runs solution
    code in a fresh `python -I` subprocess with a wall-clock timeout and
    a best-effort memory limit — it does NOT block filesystem access,
    network access, or otherwise contain a malicious payload. Only run
    code you already trust the *provenance* of (community solutions from
    the ingested corpus, or later, LLM output from a controlled prompt
    that has itself been schema-validated) — never arbitrary third-party
    submissions. For that, you'd want real containers/gVisor/seccomp.
  * Memory limits are enforced via `resource.setrlimit(RLIMIT_AS, ...)`,
    which is a POSIX-only mechanism and is unreliable on macOS in
    particular (the kernel does not consistently enforce RLIMIT_AS the
    way Linux does). Treat `memory_exceeded` as best-effort signal, not
    a guarantee. The timeout is the one limit you can actually rely on
    everywhere.
  * Only Python solutions are supported. `canonical_solution.languages`
    entries other than "python" are reported as `UNSUPPORTED_LANGUAGE`.
  * Only "plain value" problems are supported end-to-end: inputs/outputs
    that round-trip through JSON or Python literals (ints, floats,
    strings, lists, dicts, bools). Problems requiring custom structures
    (`ListNode`, `TreeNode`, etc.) will fail at the "construct the
    argument" step with a NameError from the child process — this is a
    known, deliberate scoping limit for the MVP, not a bug to chase here.
  * Comparison is exact-match after light normalization (tuples treated
    as lists). Problems that accept multiple valid answers (LeetCode's
    "if multiple answers exist, return any") will show false negatives
    here unless the example's expected output happens to match this
    solution's specific choice. Documented, not silently hidden.

Entry point detection: looks for a top-level `class Solution:` with a
public method (the first non-underscore-prefixed method, in source
order) — the shape every doocs/leetcode Python solution snippet uses —
or, failing that, a top-level public function (used by a handful of
non-OOP snippets). Anything else reports `NO_ENTRY_POINT`.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MEMORY_LIMIT_MB = 256

# Common imports LeetCode's own site auto-injects before running a
# submission, but which doocs/leetcode's README code snippets often omit
# since they're meant to be read, not executed standalone. Prepended
# ahead of the raw solution code so `List[int]`-style annotations and
# common helpers resolve without every snippet needing its own imports.
_PREAMBLE = textwrap.dedent(
    """\
    from typing import List, Optional, Tuple, Dict, Set, Any
    from collections import Counter, defaultdict, deque, OrderedDict
    from functools import lru_cache, cache, reduce, cmp_to_key
    from itertools import (
        chain, combinations, permutations, product, accumulate,
        groupby, pairwise, islice, count,
    )
    from math import inf, gcd, sqrt, ceil, floor, log2, comb, isqrt
    from bisect import bisect, bisect_left, bisect_right, insort, insort_left, insort_right
    from heapq import heappush, heappop, heapify, heappushpop, nlargest, nsmallest, merge as heapmerge
    from random import randint, random as randfloat, choice, shuffle, sample as randsample
    import heapq
    import bisect
    import math
    import itertools
    import string
    import re
    """
)

class RunOutcome(str, Enum):
    OK = "ok"  # ran to completion; check individual CaseResult.passed
    NO_ENTRY_POINT = "no_entry_point"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    NO_EXAMPLES = "no_examples"


@dataclass
class EntryPoint:
    kind: str  # "method" (Solution.<name>) or "function" (top-level <name>)
    name: str


@dataclass
class RunResult:
    """Result of executing the solution once against a single set of args."""

    value: Any = None
    error: Optional[str] = None
    timed_out: bool = False
    memory_exceeded: bool = False
    duration_seconds: float = 0.0
    stderr: str = ""


@dataclass
class CaseResult:
    input: str
    expected_raw: str
    expected_parsed: Any
    actual: Any = None
    passed: bool = False
    error: Optional[str] = None
    timed_out: bool = False
    memory_exceeded: bool = False
    duration_seconds: float = 0.0


@dataclass
class SandboxResult:
    outcome: RunOutcome
    entry_point: Optional[EntryPoint] = None
    cases: list[CaseResult] = field(default_factory=list)
    message: Optional[str] = None  # set when outcome != OK

    @property
    def all_passed(self) -> bool:
        return self.outcome == RunOutcome.OK and bool(self.cases) and all(c.passed for c in self.cases)


def detect_entry_point(code: str) -> Optional[EntryPoint]:
    """Finds the callable to invoke: the first public method on a
    top-level `class Solution`, or failing that, the first top-level
    public function. Returns None if neither is found or the code has a
    syntax error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    return EntryPoint(kind="method", name=item.name)
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            return EntryPoint(kind="function", name=node.name)
    return None


def get_entry_point_params(code: str, entry: EntryPoint) -> Optional[list[str]]:
    """Returns the ordered parameter names (excluding `self`/`cls`) of the
    entry point's function definition, or None if the code doesn't parse
    or the named entry point can't be found. Used to positionally map a
    rewritten template's renamed arguments back onto the canonical
    solver's real parameter names, since a template rewrite is free to
    rename fields (`nums` -> `transactions`) but must preserve argument
    order/count if it's a faithful rewrite.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    def _param_names(fn: ast.FunctionDef) -> list[str]:
        names = [a.arg for a in fn.args.args]
        if names and names[0] in ("self", "cls"):
            names = names[1:]
        return names

    for node in tree.body:
        if entry.kind == "method" and isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == entry.name:
                    return _param_names(item)
        if entry.kind == "function" and isinstance(node, ast.FunctionDef) and node.name == entry.name:
            return _param_names(node)
    return None


def _split_top_level(text: str) -> list[str]:
    """Splits on commas that are not nested inside any bracket/paren/brace
    depth, tracking depth by character rather than regex — needed because
    LeetCode-style inputs commonly nest lists (e.g. `intervals =
    [[1,3],[2,6]], k = 2`), and a single-level bracket-aware regex like
    `,(?![^\\[]*\\])` mis-splits as soon as nesting goes two deep.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    in_string: str | None = None  # tracks quote char if inside a string literal

    for ch in text:
        if in_string:
            current.append(ch)
            if ch == in_string:
                in_string = None
            continue
        if ch in "'\"":
            in_string = ch
            current.append(ch)
            continue
        if ch in "[({":
            depth += 1
            current.append(ch)
        elif ch in "])}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if current:
        parts.append("".join(current))

    return [p for p in parts if p.strip()]


def _parse_value(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    return text


def parse_kwargs(text: str) -> dict[str, Any]:
    """Parses a LeetCode-style example input string, e.g.
    'nums = [2,7,11,15], target = 9', into a kwargs dict.
    """
    result: dict[str, Any] = {}
    for part in _split_top_level(text):
        if "=" not in part:
            continue
        name, _, val = part.partition("=")
        result[name.strip()] = _parse_value(val.strip())
    return result


def canonicalize_value(value: Any) -> Any:
    """Normalizes tuples to lists (recursively) so a solution returning a
    tuple compares equal to a JSON-decoded list expected value. Public
    (not underscore-prefixed) since `transform/validator.py` reuses this
    exact comparison rule when checking generated variants against the
    canonical solver's output.
    """
    if isinstance(value, (list, tuple)):
        return [canonicalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: canonicalize_value(v) for k, v in value.items()}
    return value



_HARNESS_SOURCE = textwrap.dedent(
    """\
    import importlib.util
    import json
    import sys
    import traceback

    def main():
        entry_kind, entry_name, args_path, output_path, solution_path = sys.argv[1:6]

        try:
            spec = importlib.util.spec_from_file_location("solution_under_test", solution_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            with open(args_path, "r", encoding="utf-8") as f:
                args = json.load(f)

            if entry_kind == "method":
                instance = module.Solution()
                fn = getattr(instance, entry_name)
            else:
                fn = getattr(module, entry_name)

            result = fn(**args)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"ok": True, "result": result}, f)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a subprocess boundary
            # format_exception_only already includes "ErrorType: message";
            # keep only that, not a second copy of the type name.
            formatted = "".join(traceback.format_exception_only(exc)).strip()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"ok": False, "error_type": type(exc).__name__, "error": formatted}, f)
            sys.exit(1)

    if __name__ == "__main__":
        main()
    """
)


def _memory_limit_preexec(memory_limit_mb: int):
    """Returns a preexec_fn that applies a best-effort RLIMIT_AS cap.
    POSIX-only; a no-op (returns None) on Windows since preexec_fn isn't
    supported there at all.
    """
    if os.name != "posix":
        return None

    def _limit() -> None:
        try:
            import resource

            limit_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            pass  # not enforced on this platform/config; timeout is the real backstop

    return _limit


def run_code(
    solution_code: str,
    entry: EntryPoint,
    args: dict[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> RunResult:
    """Executes `entry` from `solution_code` with `args` in a fresh
    subprocess. One call = one isolated process, so a hang or crash on
    one test case can never take down the others.
    """
    with tempfile.TemporaryDirectory(prefix="sandbox_run_") as tmpdir:
        tmp = Path(tmpdir)
        solution_path = tmp / "solution_under_test.py"
        harness_path = tmp / "_harness.py"
        args_path = tmp / "args.json"
        output_path = tmp / "output.json"

        solution_path.write_text(_PREAMBLE + "\n" + solution_code, encoding="utf-8")
        harness_path.write_text(_HARNESS_SOURCE, encoding="utf-8")
        args_path.write_text(json.dumps(args), encoding="utf-8")

        cmd = [
            sys.executable,
            "-I",  # isolated mode: ignore PYTHON* env vars and user site-packages
            str(harness_path),
            entry.kind,
            entry.name,
            str(args_path),
            str(output_path),
            str(solution_path),
        ]

        start = time.monotonic()
        timed_out = False
        stderr_text = ""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=_memory_limit_preexec(memory_limit_mb),
            )
            stderr_text = proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        duration = time.monotonic() - start

        if timed_out:
            return RunResult(error="timed out", timed_out=True, duration_seconds=duration, stderr=stderr_text)

        if output_path.exists():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None
        else:
            payload = None

        if payload is None:
            # Process exited without writing a result at all — most likely
            # killed by the memory limit (SIGKILL from the kernel, no
            # Python-level exception to catch) or another hard crash.
            memory_exceeded = memory_limit_mb is not None and proc.returncode < 0
            return RunResult(
                error=f"process exited with code {proc.returncode} and produced no output"
                + (f"\nstderr: {stderr_text}" if stderr_text else ""),
                memory_exceeded=memory_exceeded,
                duration_seconds=duration,
                stderr=stderr_text,
            )

        if payload.get("ok"):
            return RunResult(value=payload.get("result"), duration_seconds=duration, stderr=stderr_text)

        error_type = payload.get("error_type", "")
        memory_exceeded = error_type == "MemoryError"
        return RunResult(
            error=payload.get("error") or error_type or "unknown error",
            memory_exceeded=memory_exceeded,
            duration_seconds=duration,
            stderr=stderr_text,
        )


def run_test_cases(
    solution_code: str,
    examples: list[dict[str, str]],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> SandboxResult:
    """Runs `solution_code` against each example (dicts with "input" and
    "output" string keys, matching CanonicalRecord.examples' shape) and
    compares actual vs expected output for each.
    """
    if not examples:
        return SandboxResult(outcome=RunOutcome.NO_EXAMPLES, message="no examples provided")

    entry = detect_entry_point(solution_code)
    if entry is None:
        return SandboxResult(
            outcome=RunOutcome.NO_ENTRY_POINT,
            message="could not find a `class Solution` method or a top-level function to call",
        )

    cases: list[CaseResult] = []
    for ex in examples:
        raw_input = ex.get("input", "")
        raw_output = ex.get("output", "")
        args = parse_kwargs(raw_input)
        expected_parsed = _parse_value(raw_output)

        run_result = run_code(solution_code, entry, args, timeout_seconds, memory_limit_mb)

        if run_result.error is not None:
            cases.append(
                CaseResult(
                    input=raw_input,
                    expected_raw=raw_output,
                    expected_parsed=expected_parsed,
                    error=run_result.error,
                    timed_out=run_result.timed_out,
                    memory_exceeded=run_result.memory_exceeded,
                    duration_seconds=run_result.duration_seconds,
                    passed=False,
                )
            )
            continue

        actual = canonicalize_value(run_result.value)
        expected = canonicalize_value(expected_parsed)
        cases.append(
            CaseResult(
                input=raw_input,
                expected_raw=raw_output,
                expected_parsed=expected_parsed,
                actual=run_result.value,
                passed=actual == expected,
                duration_seconds=run_result.duration_seconds,
            )
        )

    return SandboxResult(outcome=RunOutcome.OK, entry_point=entry, cases=cases)


def run_solution_on_examples(
    record: dict[str, Any],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> SandboxResult:
    """Convenience wrapper over `run_test_cases` that takes a canonical
    record dict (the shape found in unified_dataset.json — with
    `canonical_solution.code`/`languages` and `examples`) instead of raw
    solution code + examples.
    """
    solution = record.get("canonical_solution") or {}
    languages = solution.get("languages") or []
    code = solution.get("code")

    if not code or "python" not in [str(lang).lower() for lang in languages]:
        return SandboxResult(
            outcome=RunOutcome.UNSUPPORTED_LANGUAGE,
            message=f"no python solution code available (languages={languages})",
        )

    return run_test_cases(code, record.get("examples") or [], timeout_seconds, memory_limit_mb)
