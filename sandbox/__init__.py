"""Minimal, dev-only sandbox for executing Python solution code.

Not production-grade isolation (no containers, no seccomp, no network
denial) — see `sandbox.runner` module docstring for exact scope and
limitations. Intended for local development use: validating that a
canonical LeetCode-style solution actually produces the expected output,
and later, that LLM-generated problem variants preserve algorithmic
parity with the canonical solver.
"""
from .runner import (
    CaseResult,
    DEFAULT_MEMORY_LIMIT_MB,
    DEFAULT_TIMEOUT_SECONDS,
    EntryPoint,
    RunOutcome,
    SandboxResult,
    canonicalize_value,
    detect_entry_point,
    get_entry_point_params,
    parse_kwargs,
    run_code,
    run_solution_on_examples,
    run_test_cases,
)

__all__ = [
    "CaseResult",
    "DEFAULT_MEMORY_LIMIT_MB",
    "DEFAULT_TIMEOUT_SECONDS",
    "EntryPoint",
    "RunOutcome",
    "SandboxResult",
    "canonicalize_value",
    "detect_entry_point",
    "get_entry_point_params",
    "parse_kwargs",
    "run_code",
    "run_solution_on_examples",
    "run_test_cases",
]
