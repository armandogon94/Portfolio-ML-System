"""IEEE-CIS Fraud Detection adapter (Vesta Corporation real e-commerce payments).

Scale: ``train_transaction.csv`` is 590,540 rows x 394 columns; ``train_identity.csv``
is ~144k rows x 41 columns. About 3.5% of transactions are fraudulent.

Memory: the full transaction table in float32 is 590,540 x 393 x 4 B ~= 0.93 GB,
which is comfortable in 32 GB. Everything numeric is downcast to float32 on read
and object columns become pandas ``category``. Without that, the naive float64 +
object load is roughly 4x larger.

The competition's ``test_transaction.csv`` ships **without labels**, so it cannot
be used for evaluation. That is why the split is temporal *within* the training
file. See ``reports/RESULTS.md``.

Licence: Kaggle competition data. **Not redistributable.** No real row from this
dataset is ever committed to this repository.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_project_root

logger = logging.getLogger(__name__)

TARGET = "isFraud"
TIME_COLUMN = "TransactionDT"

PROVENANCE: dict[str, Any] = {
    "name": "IEEE-CIS Fraud Detection (Vesta)",
    "kind": "kaggle_competition",
    "slug": "ieee-fraud-detection",
    "url": "https://www.kaggle.com/competitions/ieee-fraud-detection/data",
    "licence": "Kaggle competition rules, not redistributable",
    "access": (
        "Free Kaggle account PLUS a one-click acceptance of the competition "
        "rules. Not anonymous-curl-able."
    ),
    "expected_rows": 590_540,
    "expected_positive_rate": 0.035,
    # Unrecorded because the download has not been run on this machine.
    "expected_sha256": None,
}

#: The columns the transaction table is read with. The 339 anonymised V* columns
#: are loaded too (they carry most of the signal), but they are enumerated
#: programmatically rather than listed, because their names are a contiguous range.
_TRANSACTION_BASE = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
]
_C_COLUMNS = [f"C{i}" for i in range(1, 15)]
_D_COLUMNS = [f"D{i}" for i in range(1, 16)]
_M_COLUMNS = [f"M{i}" for i in range(1, 10)]
_V_COLUMNS = [f"V{i}" for i in range(1, 340)]

#: Identity columns worth joining. The full 41 add little over these and cost a
#: wide left join on 590k rows.
_IDENTITY_COLUMNS = [
    "TransactionID",
    "id_01",
    "id_02",
    "id_05",
    "id_06",
    "id_11",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_19",
    "id_20",
    "id_30",
    "id_31",
    "id_33",
    "id_38",
    "DeviceType",
    "DeviceInfo",
]

_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_30",
    "id_31",
    "id_33",
    "id_38",
    *_M_COLUMNS,
]

#: The canonical schema. ``float32`` for every numeric column, ``category`` for
#: every string column, ``int8`` for the label.
CANONICAL_COLUMNS: dict[str, str] = {
    "TransactionID": "int64",
    "isFraud": "int8",
    "TransactionDT": "int64",
    "TransactionAmt": "float32",
    **{c: "category" for c in ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain")},
    **{
        c: "float32"
        for c in ("card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2")
    },
    **{c: "float32" for c in _C_COLUMNS},
    **{c: "float32" for c in _D_COLUMNS},
    **{c: "category" for c in _M_COLUMNS},
    **{c: "float32" for c in _V_COLUMNS},
}


def load(path: str | Path | None = None, *, sample: bool = False) -> pd.DataFrame:
    """Load IEEE-CIS in the canonical schema.

    Args:
        path: Directory containing ``train_transaction.csv`` (and optionally
            ``train_identity.csv``). When ``None``, the dataset is downloaded to
            the kagglehub cache.
        sample: Read the committed 500-row synthetic fixture instead. Used by CI
            and by the fast end-to-end test. **Never used for a published number.**

    Returns:
        DataFrame with ``isFraud`` present and ``TransactionDT`` monotonically
        usable as a split key.

    Raises:
        DatasetAccessError: The dataset could not be obtained.
        ValueError: The loaded frame has no ``isFraud`` column, i.e. it is the
            unlabelled competition test split.
    """
    if sample:
        return _load_sample()

    directory = Path(path) if path is not None else _download()
    transactions = _read_transactions(directory / "train_transaction.csv")
    identity_path = directory / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path, usecols=_IDENTITY_COLUMNS)
        transactions = transactions.merge(identity, on="TransactionID", how="left")
        logger.info("Joined train_identity.csv (%d rows)", len(identity))

    return _finalise(transactions)


def _download() -> Path:
    from src.data.download import kaggle_competition_cached

    return Path(kaggle_competition_cached(PROVENANCE["slug"]))


def _read_transactions(csv_path: Path) -> pd.DataFrame:
    """Read the transaction table with an explicit, memory-aware dtype map."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run: uv run python scripts/download_data.py --dataset ieee-cis"
        )
    wanted = [*_TRANSACTION_BASE, *_C_COLUMNS, *_D_COLUMNS, *_M_COLUMNS, *_V_COLUMNS]
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in wanted if c in header]
    dtypes: dict[Any, Any] = {
        c: "float32" for c in usecols if CANONICAL_COLUMNS.get(c) == "float32"
    }
    return pd.read_csv(csv_path, usecols=usecols, dtype=dtypes)


def _load_sample() -> pd.DataFrame:
    fixture = get_project_root() / "data" / "sample" / "ieee_cis_sample.csv"
    if not fixture.exists():
        raise FileNotFoundError(f"CI fixture missing: {fixture}")
    return _finalise(pd.read_csv(fixture))


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical dtypes and assert the label survived."""
    if TARGET not in frame.columns:
        raise ValueError(
            f"{TARGET!r} absent. The competition's test_transaction.csv is "
            "unlabelled and cannot be used for evaluation; see reports/RESULTS.md."
        )
    for column in frame.columns:
        target_dtype = CANONICAL_COLUMNS.get(column)
        if target_dtype == "category" or column in _CATEGORICAL:
            frame[column] = frame[column].astype("category")
        elif target_dtype == "float32":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    frame[TARGET] = frame[TARGET].astype("int8")
    logger.info(
        "IEEE-CIS canonical frame: %d rows x %d cols, positive rate %.4f",
        len(frame),
        frame.shape[1],
        frame[TARGET].mean(),
    )
    return frame
