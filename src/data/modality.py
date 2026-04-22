"""Modality dispatcher for multi-source data loading.

Every multi-modality model in this project routes through
:func:`load_for_modality`, which selects between three data sources:

* ``"synthetic"`` — in-repo generator (fast, deterministic, offline).
* ``"stream"``    — externally cached Kaggle dataset (real-world signal).
* ``"mixed"``     — concatenation of synthetic + stream rows, tagged with
  a ``modality`` column so downstream code can stratify, weight, or split
  by origin.

Keeping this routing in one place means new modalities only require
editing this module, and every model gets the same semantics for free.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

# Module-level import so tests can ``monkeypatch.setattr`` the name
# without importing the underlying module. During worktree bootstrap
# ``src.data.stream`` may not yet exist (it's authored by a parallel
# task); fall back to a stub that raises only if actually invoked.
try:
    from src.data.stream import kaggle_cached  # noqa: F401
except ImportError:  # pragma: no cover — only hit pre-merge in worktree

    def kaggle_cached(slug: str, *, filename: str | None = None) -> Path:
        raise ImportError(
            "src.data.stream.kaggle_cached is not available. "
            "This dispatcher depends on Task A.1.2."
        )


_VALID_MODALITIES: tuple[str, ...] = ("synthetic", "stream", "mixed")


def load_for_modality(
    modality: str,
    *,
    synthetic_loader: Callable[[], pd.DataFrame],
    stream_slug: str | None = None,
    stream_file: str | None = None,
    stream_adapter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load a DataFrame according to the requested modality.

    Args:
        modality: One of ``"synthetic"``, ``"stream"``, or ``"mixed"``.
        synthetic_loader: Zero-arg callable returning the synthetic
            DataFrame. Called only for ``"synthetic"`` and ``"mixed"``.
        stream_slug: Kaggle dataset slug (e.g. ``"owner/dataset"``).
            Required for ``"stream"`` and ``"mixed"``.
        stream_file: Filename inside the Kaggle dataset. Optional;
            passed through to :func:`kaggle_cached`.
        stream_adapter: Optional DataFrame→DataFrame transform applied
            to the raw streamed CSV (e.g. rename/select columns to
            match the synthetic schema). Identity when ``None``.

    Returns:
        For ``"synthetic"``/``"stream"``, the loaded DataFrame. For
        ``"mixed"``, a concatenation of the two with an added
        ``modality`` column whose values are ``"synthetic"`` for the
        synthetic rows and ``"stream"`` for the streamed rows.

    Raises:
        ValueError: If ``modality`` is not one of the valid values, or
            if ``"stream"``/``"mixed"`` is requested without ``stream_slug``.
    """
    if modality == "synthetic":
        return synthetic_loader()

    if modality == "stream":
        return _load_stream(stream_slug, stream_file, stream_adapter)

    if modality == "mixed":
        synthetic_df = synthetic_loader().assign(modality="synthetic")
        stream_df = _load_stream(stream_slug, stream_file, stream_adapter).assign(
            modality="stream"
        )
        return pd.concat([synthetic_df, stream_df], ignore_index=True)

    raise ValueError(
        f"Unknown modality {modality!r}. Valid modalities: {_VALID_MODALITIES}."
    )


def _load_stream(
    stream_slug: str | None,
    stream_file: str | None,
    stream_adapter: Callable[[pd.DataFrame], pd.DataFrame] | None,
) -> pd.DataFrame:
    """Fetch + read + adapt a streamed Kaggle CSV into a DataFrame."""
    if stream_slug is None:
        raise ValueError(
            "stream_slug is required for 'stream' and 'mixed' modalities."
        )

    path = kaggle_cached(stream_slug, filename=stream_file)
    df = pd.read_csv(path)
    if stream_adapter is not None:
        df = stream_adapter(df)
    return df
