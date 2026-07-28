"""Feature engineering for IEEE-CIS payment fraud.

This module is imported by **both** ``src/training/tabular.py`` and
``src/serving/preprocessing.py``. That is the whole point: training/serving skew
is the most common real ML service bug, and the only structural defence is a
single implementation. ``tests/serving/test_skew.py`` asserts the two paths
produce identical frames on a fixture batch.

Every function here is a pure function of the input frame. Nothing reads the
target and nothing uses a global. Frequency encodings are fitted on TRAIN ONLY and
passed forward as ``artifacts``: fitting them on train+test is a subtle leak.

The fitted counts and ``uid_amt_mean`` are **not point-in-time features within the
training window**: an early training row can benefit from transactions that occur
later in that same window. This avoids test leakage but still makes the offline
estimate optimistic relative to a live feature store, where only prior
transactions would exist. ``reports/RESULTS.md`` records that limitation; a full
event-time rewrite is intentionally outside this baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Columns whose value counts become features. High-cardinality identifiers where
#: "how often has this card been seen" carries more signal than the id itself.
FREQUENCY_ENCODE = ["card1", "card2", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"]

#: D1..D15 are day-deltas that drift with TransactionDT. Subtracting the
#: transaction day de-trends them; without this the model learns the calendar.
_D_COLUMNS = [f"D{i}" for i in range(1, 16)]

_SECONDS_PER_DAY = 86_400


def engineer_features(
    frame: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add engineered fraud features.

    Args:
        frame: Canonical IEEE-CIS frame (see ``src/data/adapters/ieee_cis.py``).
        artifacts: Fitted state from a previous ``fit=True`` call. Required when
            ``fit=False``.
        fit: Fit the frequency maps on this frame. Pass ``True`` for the training
            split and ``False`` for validation, test and serving.

    Returns:
        ``(engineered_frame, artifacts)``.

    Raises:
        ValueError: ``fit=False`` without artifacts. That combination silently
            produces all-NaN frequency features, so it is an error, not a default.
    """
    if not fit and artifacts is None:
        raise ValueError(
            "engineer_features(fit=False) requires the artifacts returned by the "
            "fit=True call on the training split. Passing None would emit all-NaN "
            "frequency features and quietly degrade the model."
        )

    out = frame.copy()
    artifacts = dict(artifacts or {})

    # ── Amount ────────────────────────────────────────────────────────────────
    amount = out["TransactionAmt"].astype("float32")
    out["amt_log"] = np.log1p(amount.clip(lower=0)).astype("float32")
    # The cents part of a transaction is a known IEEE-CIS signal: card-testing
    # bots produce suspiciously round amounts, humans do not.
    out["amt_decimal"] = ((amount - np.floor(amount)) * 1000).round().astype("float32")
    out["amt_is_round"] = (out["amt_decimal"] == 0).astype("int8")

    # ── Time ──────────────────────────────────────────────────────────────────
    if "TransactionDT" in out.columns:
        seconds = out["TransactionDT"].astype("int64")
        out["tx_day"] = (seconds // _SECONDS_PER_DAY).astype("int32")
        out["tx_hour"] = ((seconds // 3600) % 24).astype("int8")
        out["tx_weekday"] = ((seconds // _SECONDS_PER_DAY) % 7).astype("int8")
        # Card-testing peaks overnight. Encoded explicitly rather than left for
        # the trees to rediscover from tx_hour.
        out["tx_is_night"] = out["tx_hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")

        for column in _D_COLUMNS:
            if column in out.columns:
                out[f"{column}_detrend"] = (
                    out[column].astype("float32") - out["tx_day"].astype("float32")
                ).astype("float32")

    # ── Frequency encoding (fit on train only) ────────────────────────────────
    maps: dict[str, dict] = dict(artifacts.get("frequency_maps", {}))
    for column in FREQUENCY_ENCODE:
        if column not in out.columns:
            continue
        as_object = out[column].astype("object")
        if fit:
            maps[column] = as_object.value_counts(dropna=True).to_dict()
        out[f"{column}_freq"] = as_object.map(maps.get(column, {})).astype("float32")
    artifacts["frequency_maps"] = maps

    # ── uid-style aggregate ───────────────────────────────────────────────────
    # card1 + addr1 is the closest thing IEEE-CIS gives to a stable account key.
    if {"card1", "addr1"}.issubset(out.columns):
        uid = (
            out["card1"].astype("object").astype(str)
            + "_"
            + out["addr1"].astype("object").astype(str)
        )
        if fit:
            artifacts["uid_amt_mean"] = amount.groupby(uid).mean().to_dict()
        uid_mean = uid.map(artifacts.get("uid_amt_mean", {})).astype("float32")
        out["uid_amt_mean"] = uid_mean
        # "How unusual is this amount for this account", a ratio, so it stays
        # comparable across accounts of very different typical spend.
        out["uid_amt_ratio"] = (amount / uid_mean.replace(0, np.nan)).astype("float32")

    return out, artifacts


def get_feature_columns(frame: pd.DataFrame, denylist: list[str] | None = None) -> list[str]:
    """Return the ordered feature columns for an engineered frame.

    Selection is by exclusion: everything except the denylist and the raw
    high-cardinality identifiers that were replaced by their frequency encodings.

    Args:
        frame: A frame produced by :func:`engineer_features`.
        denylist: Columns that must never be used, from ``configs/fraud.yaml``.

    Returns:
        Column names, sorted for determinism.
    """
    blocked = set(denylist or [])
    # The raw ids are dropped in favour of *_freq; keeping both lets the model
    # memorise individual cards, which does not generalise past the split date.
    blocked.update(FREQUENCY_ENCODE)
    blocked.add("tx_day")  # a bare day index is the split boundary itself
    return sorted(c for c in frame.columns if c not in blocked)
