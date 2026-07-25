"""Feature selection for ULB credit-card fraud (OpenML 1597).

The ``V1`` through ``V28`` columns are already PCA components, so inventing
additional transformations would add complexity without meaning. ``Amount`` is
kept as supplied. ``Time`` is intentionally excluded: it is the chronological
split key, and handing the raw boundary to the model would leak dataset position.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

FEATURE_COLUMNS = [*[f"V{i}" for i in range(1, 29)], "Amount"]


def engineer_features(
    frame: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the unchanged canonical frame; ULB features need no fitted state."""
    del fit
    return frame.copy(), dict(artifacts or {})


def get_feature_columns(frame: pd.DataFrame, denylist: list[str] | None = None) -> list[str]:
    """Select PCA components plus Amount, never Time, target, or source label."""
    blocked = set(denylist or [])
    return [
        column for column in FEATURE_COLUMNS if column in frame.columns and column not in blocked
    ]
