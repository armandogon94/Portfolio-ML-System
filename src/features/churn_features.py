"""Feature engineering for credit-card customer attrition.

The raw columns are already well-scaled and semantically distinct, so this module
adds only ratios that a retention analyst would compute by hand. It deliberately
stays thin: on 10,127 rows, aggressive feature engineering is a fast route to
overfitting a dataset that is already easy.

Same shared-implementation contract as the other two feature modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def engineer_features(
    frame: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
    *,
    fit: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add engineered churn features.

    Args:
        frame: Canonical attrition frame.
        artifacts: Unused, kept for signature parity across feature modules.
        fit: Unused, same reason.

    Returns:
        ``(engineered_frame, artifacts)``.
    """
    del fit
    out = frame.copy()
    artifacts = dict(artifacts or {})

    if {"Total_Trans_Amt", "Total_Trans_Ct"}.issubset(out.columns):
        # Average ticket size. A customer whose basket shrinks while count holds
        # steady behaves differently from one who simply transacts less.
        out["avg_transaction_value"] = (
            out["Total_Trans_Amt"] / out["Total_Trans_Ct"].replace(0, np.nan)
        ).astype("float32")

    if {"Total_Revolving_Bal", "Credit_Limit"}.issubset(out.columns):
        out["revolving_utilisation"] = (
            out["Total_Revolving_Bal"] / out["Credit_Limit"].replace(0, np.nan)
        ).astype("float32")

    if {"Months_Inactive_12_mon", "Months_on_book"}.issubset(out.columns):
        out["inactive_share_of_tenure"] = (
            out["Months_Inactive_12_mon"] / out["Months_on_book"].replace(0, np.nan)
        ).astype("float32")

    if {"Contacts_Count_12_mon", "Total_Relationship_Count"}.issubset(out.columns):
        # Lots of service contacts against few products is the classic
        # "frustrated, about to leave" shape.
        out["contacts_per_product"] = (
            out["Contacts_Count_12_mon"] / out["Total_Relationship_Count"].replace(0, np.nan)
        ).astype("float32")

    return out, artifacts


def get_feature_columns(frame: pd.DataFrame, denylist: list[str] | None = None) -> list[str]:
    """Return the ordered feature columns for an engineered frame.

    The denylist from ``configs/churn.yaml`` removes ``CLIENTNUM``,
    ``Attrition_Flag`` and both ``Naive_Bayes_Classifier_*`` posterior columns.
    """
    blocked = set(denylist or [])
    return sorted(c for c in frame.columns if c not in blocked)
