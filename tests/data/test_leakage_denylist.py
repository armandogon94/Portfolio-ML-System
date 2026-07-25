"""The leakage denylist is enforced, not documented.

This is the highest-value test in the repository. A LendingClub model that sees
``recoveries`` reports ~0.99 ROC-AUC and is worthless; a churn model that sees the
``Naive_Bayes_Classifier_*`` columns reports ~1.00 and is worse than worthless.
Both are one careless column-selection change away at all times.
"""

from __future__ import annotations

import pytest

from src.data.adapters import credit_card_churn
from src.features import churn_features, credit_risk_features, fraud_features

CASES = [
    ("fraud", fraud_features, "fraud_frame"),
    ("credit_risk", credit_risk_features, "credit_risk_frame"),
    ("churn", churn_features, "churn_frame"),
]


@pytest.mark.parametrize("problem,features,frame_fixture", CASES, ids=[c[0] for c in CASES])
def test_no_denylisted_column_reaches_the_feature_matrix(problem, features, frame_fixture, request):
    config = request.getfixturevalue(f"{problem}_config")
    frame = request.getfixturevalue(frame_fixture)
    denylist = config["data"]["denylist"]

    engineered, _ = features.engineer_features(frame, fit=True)
    selected = features.get_feature_columns(engineered, denylist)

    leaked = sorted(set(denylist) & set(selected))
    assert not leaked, f"{problem}: denylisted columns selected as features: {leaked}"


@pytest.mark.parametrize("problem,features,frame_fixture", CASES, ids=[c[0] for c in CASES])
def test_the_target_is_never_a_feature(problem, features, frame_fixture, request):
    config = request.getfixturevalue(f"{problem}_config")
    frame = request.getfixturevalue(frame_fixture)

    engineered, _ = features.engineer_features(frame, fit=True)
    selected = features.get_feature_columns(engineered, config["data"]["denylist"])

    assert config["data"]["target"] not in selected


def test_churn_denylist_actually_names_the_naive_bayes_columns(churn_config, churn_frame):
    """A denylist that names the wrong string is worse than no denylist.

    The column names in the published CSV are 130+ characters long. This asserts
    the config's strings match the adapter's constants exactly, so a typo cannot
    silently disable the protection.
    """
    denylist = set(churn_config["data"]["denylist"])
    for column in credit_card_churn.NAIVE_BAYES_LEAK_COLUMNS:
        assert column in churn_frame.columns, "adapter should expose the leak columns"
        assert column in denylist, f"config denylist does not name {column[:60]}..."


def test_credit_risk_denylist_covers_every_post_outcome_field(credit_risk_config):
    """Spot-check the fields that make the model look 0.99-good and be useless."""
    denylist = set(credit_risk_config["data"]["denylist"])
    for field in (
        "recoveries",
        "collection_recovery_fee",
        "total_rec_prncp",
        "total_pymnt",
        "last_pymnt_amnt",
        "out_prncp",
        "debt_settlement_flag",
        "loan_status",
    ):
        assert field in denylist, f"{field} is post-origination and must be denylisted"
