"""Categorical schema capture and replay — the other half of skew prevention.

LightGBM encodes a pandas ``category`` column by its *codes*, and it refuses to
predict when the category set differs from the one it trained on:

    ValueError: train and valid dataset categorical_feature do not match.

That error is the friendly case. The dangerous case is when the sets happen to be
the same size but in a different order: LightGBM then predicts happily, on
silently permuted category codes. ``card4="visa"`` becomes ``card4="discover"`` and
the score is nonsense that looks plausible.

So the exact ``CategoricalDtype`` of every categorical column is captured at
training time, stored in the checkpoint alongside the feature columns, and replayed
verbatim at serving time. Both sides call the functions here — as with
``engineer_features``, one implementation is the only real defence.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import CategoricalDtype

logger = logging.getLogger(__name__)


def capture_category_dtypes(frame: pd.DataFrame) -> dict[str, list]:
    """Record the ordered category values of every categorical column.

    Stored as plain lists rather than ``CategoricalDtype`` objects so the
    checkpoint stays readable and does not pickle a pandas internal.

    Args:
        frame: The training feature matrix, after alignment.

    Returns:
        ``{column: [category, ...]}`` in the exact order LightGBM will use.
    """
    captured: dict[str, list] = {}
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, CategoricalDtype):
            captured[column] = list(dtype.categories)
    return captured


def apply_category_dtypes(frame: pd.DataFrame, captured: dict[str, list]) -> pd.DataFrame:
    """Re-impose the training category sets on a new frame.

    A value the model never saw becomes NaN, which LightGBM handles as "unknown".
    That is the correct behaviour: inventing a new code for an unseen category
    would shift every other code by one.

    Args:
        frame: A frame whose columns match the training feature columns.
        captured: The mapping returned by :func:`capture_category_dtypes`.

    Returns:
        The frame with categorical dtypes restored. Modified in place and returned.
    """
    for column, categories in captured.items():
        if column not in frame.columns:
            continue
        dtype = CategoricalDtype(categories=categories)
        before = frame[column]
        frame[column] = before.astype("object").astype(dtype)
        unseen = frame[column].isna() & before.notna()
        if unseen.any():
            logger.debug(
                "%s: %d value(s) unseen at training time -> NaN", column, int(unseen.sum())
            )
    return frame
