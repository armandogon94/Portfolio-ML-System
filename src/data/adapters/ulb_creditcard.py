"""ULB credit-card fraud adapter (OpenML dataset 1597).

OpenML requires no account, so this is the credential-free real-data fraud path.

**OpenML 1597 does not ship the ``Time`` column.** Measured on 2026-07-25:
``fetch_openml(data_id=1597, as_frame=True).frame`` returns 284,807 rows and
**30** columns -- ``V1``..``V28``, ``Amount``, ``Class`` -- with 492 positives
(0.1727%). The Kaggle mirror ``mlg-ulb/creditcardfraud`` does carry ``Time``,
but that path needs an account, which defeats the point of this adapter.

The consequence is methodological and is stated rather than papered over: with
no timestamp, a chronological split is impossible on this source, so
``configs/fraud_ulb.yaml`` uses stratified k-fold. Row *position* is not used as
a time proxy: OpenML's ordering is not documented as chronological and this
repository does not assert facts it has not verified. The temporal-split
demonstration lives on the IEEE-CIS path (``configs/fraud.yaml``).

``Time`` is accepted if a caller supplies a source-shaped CSV that has it (the
Kaggle mirror), in which case it is carried through as a split key and the
feature module still excludes it from the model matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_root

logger = logging.getLogger(__name__)

TARGET = "is_fraud"
#: Present only in the Kaggle mirror, never in OpenML 1597. Optional throughout.
TIME_COLUMN = "Time"
_V_COLUMNS = [f"V{i}" for i in range(1, 29)]
#: Columns OpenML 1597 always provides. ``Time`` is deliberately not among them.
_REQUIRED_NUMERIC = [*_V_COLUMNS, "Amount"]
_OPTIONAL_NUMERIC = [TIME_COLUMN]

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
    # Unrecorded because the download has not been run on this machine.
    "expected_sha256": None,
}

CANONICAL_COLUMNS: dict[str, str] = {
    TARGET: "int8",
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
    missing = [column for column in _REQUIRED_NUMERIC if column not in frame.columns]
    if missing:
        raise ValueError(
            f"ULB source is missing required columns: {missing}. Fetch OpenML data_id 1597 "
            "or pass its source-shaped CSV."
        )
    present = [*_REQUIRED_NUMERIC, *(c for c in _OPTIONAL_NUMERIC if c in frame.columns)]

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
        "present" if TIME_COLUMN in canonical.columns else "absent - OpenML omits it",
        canonical[TARGET].mean(),
    )
    return canonical
