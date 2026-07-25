"""Adapters map raw vendor columns to a stable canonical schema."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.adapters import ADAPTERS, credit_card_churn, get_adapter, ieee_cis, lending_club

ALL_ADAPTERS = [ieee_cis, lending_club, credit_card_churn]


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_adapter_satisfies_the_contract(adapter):
    """Every adapter exposes the three names tabular.py depends on."""
    assert isinstance(adapter.CANONICAL_COLUMNS, dict)
    assert adapter.CANONICAL_COLUMNS, "canonical schema must not be empty"
    assert callable(adapter.load)
    for key in ("name", "kind", "url", "licence", "access"):
        assert key in adapter.PROVENANCE, f"{adapter.__name__} PROVENANCE missing {key!r}"


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_sample_load_yields_the_target_as_int8_binary(adapter):
    frame = adapter.load(sample=True)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) > 0
    target = frame[adapter.TARGET]
    assert str(target.dtype) == "int8"
    assert set(target.unique()) <= {0, 1}


@pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_no_column_is_entirely_null(adapter):
    """An all-null column means the fixture or the dtype cast is broken."""
    frame = adapter.load(sample=True)
    all_null = [c for c in frame.columns if frame[c].isna().all()]
    assert not all_null, f"{adapter.__name__} produced all-null columns: {all_null}"


def test_ieee_cis_rejects_an_unlabelled_frame():
    """The competition's test split has no isFraud. Loading it must fail loudly."""
    unlabelled = ieee_cis.load(sample=True).drop(columns=["isFraud"])
    with pytest.raises(ValueError, match="isFraud"):
        ieee_cis._finalise(unlabelled)


def test_lending_club_drops_non_terminal_statuses():
    """Current loans have no outcome and must never be labelled as non-default."""
    raw = pd.read_csv(lending_club._fixture())
    assert (raw["loan_status"] == "Current").any(), "fixture should contain Current rows"

    loaded = lending_club.load(sample=True)
    assert len(loaded) < len(raw), "non-terminal rows were not dropped"
    assert set(loaded["is_default"].unique()) <= {0, 1}


def test_churn_keeps_the_naive_bayes_columns_for_the_leak_demo():
    """The two posterior columns must survive the adapter so a test can remove them."""
    frame = credit_card_churn.load(sample=True)
    for column in credit_card_churn.NAIVE_BAYES_LEAK_COLUMNS:
        assert column in frame.columns


def test_churn_rejects_unknown_attrition_values():
    frame = pd.read_csv(credit_card_churn._fixture())
    frame.loc[0, "Attrition_Flag"] = "Maybe Customer"
    with pytest.raises(ValueError, match="Unexpected Attrition_Flag"):
        credit_card_churn._finalise(frame)


def test_get_adapter_resolves_every_registered_path():
    for problem, path in ADAPTERS.items():
        module = get_adapter(path)
        assert hasattr(module, "load"), problem


def test_get_adapter_rejects_a_non_adapter_module():
    with pytest.raises(KeyError, match="not a valid adapter"):
        get_adapter("src.config")
