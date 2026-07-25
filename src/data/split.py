"""Data splitting. Time-based by default, because these are all fintech problems.

A random split on IEEE-CIS puts the same card and device on both sides of the
boundary; on LendingClub it mixes 2012 and 2018 vintages. Both inflate the score
in a way that does not survive deployment. ``tests/training/test_split.py``
asserts no later timestamp lands in train than the earliest in test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Row-index arrays for one train/validation/test partition.

    ``val`` may be empty when the config asks for no validation fold.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def describe(self) -> dict[str, int]:
        return {"n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test)}


def time_split(
    frame: pd.DataFrame,
    column: str,
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Split:
    """Chronological split: earliest rows train, latest rows test.

    Args:
        frame: The full dataset.
        column: The ordering key (``TransactionDT``, ``issue_d``).
        test_size: Tail fraction held out as test.
        val_size: Fraction of the *whole* dataset carved from the tail of train
            as an early-stopping fold. Also chronological, so validation sits
            between train and test in time.

    Returns:
        A :class:`Split`.

    Raises:
        KeyError: ``column`` is absent.
        ValueError: The requested fractions leave no training rows.
    """
    if column not in frame.columns:
        raise KeyError(
            f"split column {column!r} not in frame. Columns: {sorted(frame.columns)[:20]}..."
        )
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {test_size + val_size}")

    order = np.argsort(frame[column].to_numpy(), kind="stable")
    n = len(order)
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_test - n_val
    if n_train <= 0:
        raise ValueError(f"No training rows left: n={n}, test={n_test}, val={n_val}")

    return Split(
        train=order[:n_train],
        val=order[n_train : n_train + n_val],
        test=order[n_train + n_val :],
    )


def stratified_kfold_indices(
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[Split]:
    """Stratified K-fold splits, for datasets too small for a single hold-out.

    Used by the churn problem (n = 10,127). Every metric derived from these folds
    is reported as mean +/- std; a single fold number on 10k rows is noise.

    Args:
        y: Binary labels.
        n_splits: Number of folds.
        seed: Shuffle seed.

    Returns:
        One :class:`Split` per fold, with an empty ``val`` array — early stopping
        is disabled for this problem, the trees are few and fixed.
    """
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        Split(train=train_idx, val=np.array([], dtype=int), test=test_idx)
        for train_idx, test_idx in splitter.split(np.zeros(len(labels)), labels)
    ]


def make_splits(frame: pd.DataFrame, split_config: dict, target: str, seed: int) -> list[Split]:
    """Dispatch on ``split.type`` from the config.

    Returns:
        A list of splits. Length 1 for ``time`` and ``random``, ``n_splits`` for
        ``stratified_kfold``. The trainer always iterates, so both shapes take
        the same code path.

    Raises:
        ValueError: Unknown ``split.type``.
    """
    kind = split_config["type"]

    if kind == "time":
        return [
            time_split(
                frame,
                split_config["column"],
                test_size=split_config.get("test_size", 0.2),
                val_size=split_config.get("val_size", 0.0),
            )
        ]

    if kind == "stratified_kfold":
        return stratified_kfold_indices(
            frame[target], n_splits=split_config.get("n_splits", 5), seed=seed
        )

    if kind == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        n_test = int(round(len(frame) * split_config.get("test_size", 0.2)))
        return [Split(train=order[n_test:], val=np.array([], dtype=int), test=order[:n_test])]

    raise ValueError(f"Unknown split.type {kind!r}. Expected time | stratified_kfold | random.")
