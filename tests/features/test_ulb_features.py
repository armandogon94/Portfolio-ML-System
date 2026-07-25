"""ULB feature selection keeps PCA components and excludes the time split key."""

from __future__ import annotations

from src.data.adapters import ulb_creditcard
from src.features import ulb_features


def test_ulb_selects_only_pca_components_and_amount():
    frame = ulb_creditcard.load(sample=True)
    engineered, _ = ulb_features.engineer_features(frame, fit=True)
    selected = ulb_features.get_feature_columns(engineered, ["is_fraud", "Class"])

    assert selected == [*[f"V{i}" for i in range(1, 29)], "Amount"]
    assert "Time" not in selected
    assert "is_fraud" not in selected
