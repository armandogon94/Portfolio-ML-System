"""LendingClub accepted loans 2007-2018Q4 adapter.

Source file: ``accepted_2007_to_2018Q4.csv.gz`` — ~2.26M rows x 151 columns,
~648 MB gzipped. **Never read all 151 columns.** ``usecols`` keeps the load to the
~30 origination-time fields the model is allowed to see, which is the difference
between a 3 GB frame and a 300 MB one.

Target construction: ``loan_status`` filtered to TERMINAL outcomes only.

    "Fully Paid"  -> is_default = 0
    "Charged Off" -> is_default = 1

Every other status (``Current``, ``In Grace Period``, ``Late (...)``,
``Default``, ``Issued``) is **dropped**, because the loan has not resolved. Keeping
``Current`` and calling it non-default labels an unresolved loan as a success and
biases the model toward optimism on young vintages.

Licence: the uploader tags CC0, but upstream authority is unverified. Do not
redistribute rows. Kaggle account required; no rules gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_root

logger = logging.getLogger(__name__)

TARGET = "is_default"
TIME_COLUMN = "issue_d"

PROVENANCE: dict[str, Any] = {
    "name": "LendingClub accepted loans 2007-2018Q4",
    "kind": "kaggle_dataset",
    "slug": "wordsforthewise/lending-club",
    "filename": "accepted_2007_to_2018Q4.csv.gz",
    "url": "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
    "licence": ("Uploader tags CC0; upstream authority unverified — do not redistribute rows"),
    "access": "Free Kaggle account. No rules gate.",
    "expected_rows": 2_260_701,
    "expected_rows_raw": 2_260_701,
    # Unrecorded because the download has not been run on this machine.
    "expected_sha256": None,
    "note": "Row count after filtering to terminal statuses is recorded in data/README.md",
}

#: Terminal loan statuses and their label. Anything not in this map is dropped.
TERMINAL_STATUS = {
    "Fully Paid": 0,
    "Charged Off": 1,
}

#: Origination-time columns only. Every post-origination field lives in the
#: denylist in ``configs/credit_risk.yaml`` and is additionally never read here.
_USECOLS = [
    "loan_status",
    "issue_d",
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

_CATEGORICAL = [
    "term",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

_NUMERIC = [
    "loan_amnt",
    "funded_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
]

CANONICAL_COLUMNS: dict[str, str] = {
    "is_default": "int8",
    "issue_d": "datetime64[ns]",
    "earliest_cr_line": "datetime64[ns]",
    **{c: "category" for c in _CATEGORICAL},
    **{c: "float32" for c in _NUMERIC},
}


def load(path: str | Path | None = None, *, sample: bool = False) -> pd.DataFrame:
    """Load LendingClub in the canonical schema, terminal statuses only.

    Args:
        path: Path to ``accepted_2007_to_2018Q4.csv.gz`` (or a directory holding
            it). When ``None``, the dataset is downloaded to the kagglehub cache.
        sample: Read the committed 500-row synthetic fixture instead.

    Returns:
        DataFrame with ``is_default`` in {0, 1} and a parsed ``issue_d``.
    """
    if sample:
        return _finalise(pd.read_csv(_fixture()))

    source = Path(path) if path is not None else _download()
    if source.is_dir():
        matches = sorted(source.rglob(PROVENANCE["filename"]))
        if not matches:
            raise FileNotFoundError(
                f"{PROVENANCE['filename']} not found under {source}. Run: "
                "uv run python scripts/download_data.py --dataset lending-club"
            )
        source = matches[0]

    header = pd.read_csv(source, nrows=0).columns.tolist()
    usecols = [c for c in _USECOLS if c in header]
    frame = pd.read_csv(source, usecols=usecols, low_memory=False)
    logger.info("Read %d raw LendingClub rows", len(frame))
    return _finalise(frame)


def _download() -> Path:
    from src.data.download import kaggle_dataset_cached

    return Path(kaggle_dataset_cached(PROVENANCE["slug"]))


def _fixture() -> Path:
    fixture = get_project_root() / "data" / "sample" / "lending_club_sample.csv"
    if not fixture.exists():
        raise FileNotFoundError(f"CI fixture missing: {fixture}")
    return fixture


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter to terminal statuses, derive the label, apply canonical dtypes."""
    if "loan_status" not in frame.columns:
        raise ValueError("loan_status absent — cannot derive is_default without it.")

    before = len(frame)
    frame = frame[frame["loan_status"].isin(TERMINAL_STATUS)].copy()
    frame[TARGET] = frame["loan_status"].map(TERMINAL_STATUS).astype("int8")
    logger.info(
        "Filtered to terminal statuses: %d -> %d rows (%.1f%% dropped as unresolved)",
        before,
        len(frame),
        100.0 * (1 - len(frame) / before) if before else 0.0,
    )

    for column in ("issue_d", "earliest_cr_line"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], format="%b-%Y", errors="coerce")

    for column in _NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")

    for column in _CATEGORICAL:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")

    if frame[TARGET].nunique() < 2:
        raise ValueError(
            "is_default has a single class after filtering — the source file is "
            "probably truncated or the wrong split."
        )

    logger.info(
        "LendingClub canonical frame: %d rows x %d cols, default rate %.4f",
        len(frame),
        frame.shape[1],
        frame[TARGET].mean(),
    )
    return frame.reset_index(drop=True)
