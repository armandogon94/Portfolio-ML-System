"""Credit-card customer attrition adapter (``sakshigoyal7/credit-card-customers``).

The public run record measures 10,127 rows and a 16.07% attrition rate. Evaluation
uses five-fold stratified cross-validation rather than a single hold-out split.

Two columns in the published CSV are traps. The dataset ships
``Naive_Bayes_Classifier_Attrition_Flag_...1`` and ``...2``, which are pre-computed
posterior probabilities of the target, which the dataset's own author tells you to
delete. They are the label, laundered through a classifier. This adapter keeps
them in the returned frame *on purpose* so a future controlled comparison is
possible; that comparison is not measured. The denylist in ``configs/churn.yaml``
is what stops them reaching the published model.

Licence: Kaggle dataset, free account, no rules gate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_root

logger = logging.getLogger(__name__)

TARGET = "is_attrited"

PROVENANCE: dict[str, Any] = {
    "name": "Credit Card Customers (bank attrition)",
    "kind": "kaggle_dataset",
    "slug": "sakshigoyal7/credit-card-customers",
    "filename": "BankChurners.csv",
    "url": "https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers",
    "licence": ("Uploader tags CC0; upstream authority unverified. Do not redistribute rows"),
    "access": "Free Kaggle account. No rules gate.",
    "expected_rows": 10_127,
    "expected_positive_rate": 0.1607,
    # Measured from Kaggle dataset version 1 on 2026-07-30.
    "expected_sha256": "c91b525a2a6755a1b0b80dad1d0d008ca97ec4df34552c8f47ffa12b6184b779",
}

#: The two pre-computed posterior columns. Present in the frame, excluded by the
#: config denylist. Named here so a test can assert they were not silently dropped.
NAIVE_BAYES_LEAK_COLUMNS = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_"
    "Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_"
    "Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

_CATEGORICAL = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

_NUMERIC = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

CANONICAL_COLUMNS: dict[str, str] = {
    "is_attrited": "int8",
    **{c: "category" for c in _CATEGORICAL},
    **{c: "float32" for c in _NUMERIC},
}

#: ``Attrition_Flag`` values mapped to the binary label.
ATTRITION_MAP = {
    "Attrited Customer": 1,
    "Existing Customer": 0,
}


def load(path: str | Path | None = None, *, sample: bool = False) -> pd.DataFrame:
    """Load the attrition dataset in the canonical schema.

    Args:
        path: Path to ``BankChurners.csv`` or a directory holding it. When
            ``None``, the dataset is downloaded to the kagglehub cache.
        sample: Read the committed 500-row synthetic fixture instead.

    Returns:
        DataFrame with ``is_attrited`` in {0, 1}.
    """
    if sample:
        return _finalise(pd.read_csv(_fixture()))

    source = Path(path) if path is not None else _download()
    if source.is_dir():
        matches = sorted(source.rglob(PROVENANCE["filename"]))
        if not matches:
            raise FileNotFoundError(
                f"{PROVENANCE['filename']} not found under {source}. Run: "
                "uv run python scripts/download_data.py --dataset cc-churn"
            )
        source = matches[0]

    return _finalise(pd.read_csv(source))


def _download() -> Path:
    from src.data.download import kaggle_dataset_cached

    return Path(kaggle_dataset_cached(PROVENANCE["slug"]))


def _fixture() -> Path:
    fixture = get_project_root() / "data" / "sample" / "churn_sample.csv"
    if not fixture.exists():
        raise FileNotFoundError(f"CI fixture missing: {fixture}")
    return fixture


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the label and apply canonical dtypes."""
    if "Attrition_Flag" not in frame.columns:
        raise ValueError("Attrition_Flag absent: cannot derive is_attrited without it.")

    unknown = set(frame["Attrition_Flag"].unique()) - set(ATTRITION_MAP)
    if unknown:
        raise ValueError(f"Unexpected Attrition_Flag values: {sorted(unknown)}")

    frame = frame.copy()
    frame[TARGET] = frame["Attrition_Flag"].map(ATTRITION_MAP).astype("int8")

    # Kaggle exports of this file carry a trailing unnamed index column.
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]

    for column in _NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    for column in _CATEGORICAL:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")

    logger.info(
        "Attrition canonical frame: %d rows x %d cols, attrition rate %.4f",
        len(frame),
        frame.shape[1],
        frame[TARGET].mean(),
    )
    return frame.reset_index(drop=True)
