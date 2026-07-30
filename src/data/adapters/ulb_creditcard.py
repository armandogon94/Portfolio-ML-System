"""ULB credit-card fraud adapter (OpenML dataset 1597).

OpenML requires no account, so this is the credential-free real-data fraud path.

OpenML documents ``Time`` as the row-id attribute and as seconds since the first
transaction. Scikit-learn omits row-id attributes from ``fetch_openml().frame``;
the download layer restores that column from the exact cached ARFF rather than
mistaking client-library behavior for a source limitation. The config uses
``Time`` only as the chronological split key, never as a model feature.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_root

logger = logging.getLogger(__name__)

TARGET = "is_fraud"
TIME_COLUMN = "Time"
_V_COLUMNS = [f"V{i}" for i in range(1, 29)]
_REQUIRED_NUMERIC = [*_V_COLUMNS, "Amount"]

PROVENANCE: dict[str, Any] = {
    "name": "ULB Credit Card Fraud (OpenML id 1597)",
    "kind": "openml",
    "data_id": 1597,
    "url": "https://www.openml.org/d/1597",
    "licence": (
        'Unresolved: OpenML records only "Public"; the Kaggle mirror indicates '
        "ODbL-style terms. Treat as NOT cleared for redistribution."
    ),
    "access": "NO ACCOUNT REQUIRED. Fetched via sklearn.datasets.fetch_openml.",
    "expected_rows": 284_807,
    "expected_positive_rate": 0.001727,
    "expected_md5": "178bcf9bb1f31a3dfe12d0e577884add",
    "expected_sha256": None,
}

CANONICAL_COLUMNS: dict[str, str] = {
    TARGET: "int8",
    TIME_COLUMN: "float32",
    **{column: "float32" for column in _REQUIRED_NUMERIC},
}


def load(path: str | Path | None = None, *, sample: bool = False) -> pd.DataFrame:
    """Load OpenML 1597 in the canonical schema.

    Args:
        path: Optional local source-shaped CSV for offline inspection. When
            omitted, the real path calls :func:`src.data.download.openml_cached`
            with dataset id 1597.
        sample: Read the repository's 500-row synthetic fixture instead. It
            contains no real ULB rows and is never used for a published number.

    Returns:
        A frame with an ``int8`` ``is_fraud`` target and float32 feature/split
        columns.
    """
    if sample:
        frame = pd.read_csv(_fixture())
    elif path is not None:
        frame = pd.read_csv(Path(path))
    else:
        from src.data.download import openml_cached

        frame = openml_cached(PROVENANCE["data_id"])
    return _finalise(frame)


def _fixture() -> Path:
    fixture = get_project_root() / "data" / "sample" / "ulb_creditcard_sample.csv"
    if not fixture.exists():
        raise FileNotFoundError(f"CI fixture missing: {fixture}")
    return fixture


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive ``is_fraud`` from ``Class`` and enforce the canonical dtypes."""
    required = [TIME_COLUMN, *_REQUIRED_NUMERIC]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"ULB source is missing required columns: {missing}. Fetch OpenML data_id 1597 "
            "or pass its source-shaped CSV."
        )
    present = required

    label_column = "Class" if "Class" in frame.columns else TARGET
    if label_column not in frame.columns:
        raise ValueError(
            "ULB source has no 'Class' label, so is_fraud cannot be derived. "
            "Fetch OpenML data_id 1597 rather than an unlabelled derivative."
        )
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    if labels.isna().any() or not set(labels.unique()) <= {0, 1}:
        raise ValueError("ULB Class must contain only binary 0/1 labels.")

    canonical = frame[present].copy()
    for column in present:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce").astype("float32")
    canonical.insert(0, TARGET, labels.astype("int8"))

    logger.info(
        "ULB canonical frame: %d rows x %d cols (Time %s), fraud rate %.6f",
        len(canonical),
        canonical.shape[1],
        "present",
        canonical[TARGET].mean(),
    )
    return canonical
