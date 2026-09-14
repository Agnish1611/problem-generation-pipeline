"""Embedding generation and caching for algorithmic problems.

Uses sentence-transformers (all-MiniLM-L6-v2) to convert canonical
problem records into dense semantic vectors, with atomic caching to disk.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = "data/processed"

_MODEL_CACHE: dict[str, Any] = {}


def get_sentence_transformer(model_name: str = DEFAULT_MODEL_NAME):
    """Lazy-load and cache the SentenceTransformer model."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence transformer model: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def compose_text(problem: dict[str, Any]) -> str:
    """Compose text representation of a problem for embedding:

    Format: {title}\\n{description[:2000]}\\n{examples_sample}
    where examples_sample concatenates up to 2 examples.
    """
    title = str(problem.get("title") or "").strip()
    desc = str(problem.get("description") or "")[:2000].strip()

    examples = problem.get("examples") or []
    ex_parts = []
    for ex in examples[:2]:
        if isinstance(ex, dict):
            inp = ex.get("input", "")
            out = ex.get("output", "")
            ex_parts.append(f"Input: {inp}\nOutput: {out}")
        elif isinstance(ex, str):
            ex_parts.append(ex)
    examples_sample = "\n".join(ex_parts).strip()

    parts = [title]
    if desc:
        parts.append(desc)
    if examples_sample:
        parts.append(examples_sample)
    return "\n".join(parts)


def get_text_embedding(
    problem: dict[str, Any],
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> np.ndarray:
    """Compute 1D normalized float32 semantic embedding for a canonical problem dict."""
    text = compose_text(problem)
    if model is None:
        model = get_sentence_transformer(model_name)
    assert model is not None
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def _atomic_write_npy(filepath: Path, arr: np.ndarray) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_name(f".tmp_{os.getpid()}_{filepath.name}")
    try:
        np.save(tmp_path, arr)
        # np.save ensures .npy extension is present
        actual_saved = tmp_path if str(tmp_path).endswith(".npy") else tmp_path.with_suffix(".npy")
        os.replace(actual_saved, filepath)
    except Exception:
        for p in [tmp_path, tmp_path.with_suffix(".npy")]:
            if p.exists():
                p.unlink()
        raise


def _atomic_write_json(filepath: Path, data: Any) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_name(f".tmp_{os.getpid()}_{filepath.name}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_embeddings(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Tuple[np.ndarray, List[str]]:
    """Load cached embeddings and corresponding problem IDs."""
    cache_path = Path(cache_dir)
    emb_file = cache_path / "embeddings.npy"
    ids_file = cache_path / "ids.json"

    if not emb_file.exists() or not ids_file.exists():
        raise FileNotFoundError(
            f"Embedding cache not found at {emb_file} and {ids_file}. "
            "Call compute_all_embeddings() first."
        )

    embeddings = np.load(emb_file)
    with open(ids_file, "r", encoding="utf-8") as f:
        ids = json.load(f)

    if len(embeddings) != len(ids):
        raise ValueError(
            f"Embeddings length ({len(embeddings)}) mismatch with IDs length ({len(ids)})."
        )

    logger.info("Loaded %d embeddings of shape %s from %s", len(ids), embeddings.shape, cache_dir)
    return embeddings, ids


def compute_all_embeddings(
    problems: Optional[List[dict[str, Any]]] = None,
    unified_json_path: str | Path = "data/output/unified_dataset.json",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    recompute: bool = False,
    batch_size: int = 64,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[np.ndarray, List[str]]:
    """Compute embeddings for all valid problems and cache to disk.

    - Validates problems (skips records lacking 'id' or 'title', logs to skipped.json).
    - If cache exists and recompute=False, loads and returns cached embeddings.
    - Uses safe atomic writes for data/processed/embeddings.npy and ids.json.
    """
    cache_path = Path(cache_dir)
    emb_file = cache_path / "embeddings.npy"
    ids_file = cache_path / "ids.json"
    skipped_file = cache_path / "skipped.json"

    if not recompute and emb_file.exists() and ids_file.exists():
        logger.info("Existing embeddings cache found at %s. Loading...", cache_dir)
        return load_embeddings(cache_dir=cache_path)

    if problems is None:
        logger.info("Loading unified dataset from %s", unified_json_path)
        with open(unified_json_path, "r", encoding="utf-8") as f:
            problems = json.load(f)
    assert problems is not None

    valid_problems: List[dict[str, Any]] = []
    skipped_problems: List[dict[str, Any]] = []

    for p in problems:
        pid = p.get("id")
        title = p.get("title")
        if not pid or not title or not str(title).strip():
            skipped_problems.append({
                "problem": p,
                "reason": "Missing id or title",
            })
        else:
            valid_problems.append(p)

    if skipped_problems:
        logger.warning(
            "Skipping %d problems lacking 'id' or 'title'. Logging to %s",
            len(skipped_problems),
            skipped_file,
        )
        _atomic_write_json(skipped_file, skipped_problems)
    elif skipped_file.exists():
        # Clear out previous skipped file if none skipped
        _atomic_write_json(skipped_file, [])

    if not valid_problems:
        logger.warning("No valid problems found to compute embeddings.")
        empty_emb = np.empty((0, 384), dtype=np.float32)
        _atomic_write_npy(emb_file, empty_emb)
        _atomic_write_json(ids_file, [])
        return empty_emb, []

    texts = [compose_text(p) for p in valid_problems]
    ids = [str(p["id"]) for p in valid_problems]

    if model is None:
        model = get_sentence_transformer(model_name)
    assert model is not None

    logger.info("Encoding %d problem texts in batches of %d...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    logger.info("Writing embeddings (shape %s) to %s atomically", embeddings.shape, emb_file)
    _atomic_write_npy(emb_file, embeddings)
    _atomic_write_json(ids_file, ids)

    return embeddings, ids
