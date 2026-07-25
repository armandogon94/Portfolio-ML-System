"""Download layer: credential handling, competition path, actionable errors.

Every test here mocks kagglehub. The single real-network canary is marked
``network`` and deselected in CI explicitly — see .github/workflows/ci.yml, not a
hidden addopts setting.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.download import (
    DatasetAccessError,
    kaggle_competition_cached,
    kaggle_dataset_cached,
    sha256_of,
)


@pytest.fixture
def no_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))


@pytest.fixture
def fake_credentials(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "deadbeef")


def test_missing_credentials_names_the_exact_remediation(no_credentials):
    with pytest.raises(DatasetAccessError) as excinfo:
        kaggle_dataset_cached("owner/slug")
    message = str(excinfo.value)
    assert "kaggle.com/settings/account" in message
    assert "~/.kaggle/kaggle.json" in message
    # The absence of a synthetic fallback is a deliberate, stated policy.
    assert "No synthetic fallback" in message


def test_competition_failure_points_at_the_rules_page(fake_credentials):
    """A 403 on a competition almost always means unaccepted rules."""
    fake = MagicMock()
    fake.competition_download.side_effect = RuntimeError("403 Forbidden")
    with patch.dict("sys.modules", {"kagglehub": fake}):
        with pytest.raises(DatasetAccessError) as excinfo:
            kaggle_competition_cached("ieee-fraud-detection")
    message = str(excinfo.value)
    assert "competitions/ieee-fraud-detection/rules" in message
    assert "I Understand and Accept" in message


def test_competition_download_is_used_for_competitions(fake_credentials, tmp_path):
    """dataset_download cannot fetch a competition — the old code only had that."""
    fake = MagicMock()
    fake.competition_download.return_value = str(tmp_path)
    with patch.dict("sys.modules", {"kagglehub": fake}):
        result = kaggle_competition_cached("ieee-fraud-detection")
    fake.competition_download.assert_called_once_with("ieee-fraud-detection")
    fake.dataset_download.assert_not_called()
    assert result == tmp_path


def test_dataset_download_is_used_for_datasets(fake_credentials, tmp_path):
    fake = MagicMock()
    fake.dataset_download.return_value = str(tmp_path)
    with patch.dict("sys.modules", {"kagglehub": fake}):
        result = kaggle_dataset_cached("wordsforthewise/lending-club")
    fake.dataset_download.assert_called_once()
    assert result == tmp_path


def test_missing_named_file_lists_what_is_actually_there(fake_credentials, tmp_path):
    (tmp_path / "something_else.csv").write_text("a,b\n1,2\n")
    fake = MagicMock()
    fake.dataset_download.return_value = str(tmp_path)
    with patch.dict("sys.modules", {"kagglehub": fake}):
        with pytest.raises(DatasetAccessError, match="something_else.csv"):
            kaggle_dataset_cached("owner/slug", filename="expected.csv")


def test_nested_file_is_found(fake_credentials, tmp_path):
    """kagglehub sometimes nests one level deeper than the slug implies."""
    nested = tmp_path / "versions" / "1"
    nested.mkdir(parents=True)
    (nested / "BankChurners.csv").write_text("a\n1\n")
    fake = MagicMock()
    fake.dataset_download.return_value = str(tmp_path)
    with patch.dict("sys.modules", {"kagglehub": fake}):
        found = kaggle_dataset_cached("owner/slug", filename="BankChurners.csv")
    assert found.name == "BankChurners.csv"


def test_sha256_matches_a_known_value(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert sha256_of(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


@pytest.mark.network
def test_real_kaggle_canary():
    """The one test that hits Kaggle. Deselected in CI with -m 'not network'."""
    path = kaggle_dataset_cached("sakshigoyal7/credit-card-customers")
    assert path.exists()
