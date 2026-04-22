"""Streaming data loaders for Kaggle and Hugging Face datasets.

Keeps external datasets out of the repo by either caching to
``~/.cache/kagglehub/`` (kagglehub) or streaming rows in-memory
(HF ``datasets`` with ``streaming=True``).

Heavy imports (``kagglehub``, ``datasets``) happen inside functions
so this module can be imported cheaply and mocked cleanly in tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def kaggle_cached(slug: str, *, filename: str | None = None) -> Path:
    """Download a Kaggle dataset to ``~/.cache/kagglehub/`` and return the path.

    Args:
        slug: Kaggle dataset slug, e.g. ``"arianazmoudeh/airbnbopendata"``.
        filename: Optional specific file inside the dataset. When given,
            the returned path points to that file; otherwise to the
            dataset directory.

    Returns:
        ``pathlib.Path`` to the cached dataset (directory or file).

    Raises:
        RuntimeError: If Kaggle credentials are missing.
    """
    # Imported here so unit tests can mock without a real network/creds.
    from src.data.kaggle_credentials import ensure_kaggle_env

    ensure_kaggle_env()

    import kagglehub

    cache_dir = Path(kagglehub.dataset_download(slug))
    logger.info("Kaggle dataset %s cached at %s", slug, cache_dir)

    return cache_dir / filename if filename else cache_dir


def hf_stream(dataset_id: str, split: str = "train") -> Iterator[dict[str, Any]]:
    """Stream rows from a Hugging Face dataset without downloading.

    Wraps ``datasets.load_dataset(..., streaming=True)`` and yields each
    row as a plain dict. Useful for large datasets that would bloat disk
    if downloaded in full.

    Args:
        dataset_id: HF dataset identifier, e.g. ``"lex_glue"``.
        split: Dataset split name. Defaults to ``"train"``.

    Yields:
        One dict per row.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    logger.debug("Opened HF streaming dataset %s[%s]", dataset_id, split)
    yield from ds


def iter_batches(
    source: pd.DataFrame | Iterable[dict[str, Any]],
    batch_size: int = 1024,
) -> Iterator[pd.DataFrame]:
    """Chunk a DataFrame or iterator of dicts into ``batch_size`` DataFrames.

    Args:
        source: Either a ``pandas.DataFrame`` or an iterable yielding dicts
            (e.g. the return value of :func:`hf_stream`).
        batch_size: Maximum rows per emitted DataFrame.

    Yields:
        ``pandas.DataFrame`` chunks. The final chunk may be smaller than
        ``batch_size``. Empty sources yield nothing.
    """
    if isinstance(source, pd.DataFrame):
        for start in range(0, len(source), batch_size):
            yield source.iloc[start : start + batch_size].reset_index(drop=True)
        return

    batch: list[dict[str, Any]] = []
    for row in source:
        batch.append(row)
        if len(batch) >= batch_size:
            yield pd.DataFrame(batch)
            batch = []
    if batch:
        yield pd.DataFrame(batch)
