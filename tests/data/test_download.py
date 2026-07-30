"""Download layer: credential handling, competition path, actionable errors.

Every test here mocks kagglehub. The single real-network canary is marked
``network`` and deselected in CI explicitly; see .github/workflows/ci.yml, not a
hidden addopts setting.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts import download_data
from src.data import download
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
    assert "~/.kaggle/access_token" in message
    assert "~/.kaggle/kaggle.json" in message


def test_competition_download_is_used_for_competitions(fake_credentials, tmp_path):
    """dataset_download cannot fetch a competition, and the old code only had that."""
    fake = MagicMock()
    fake.competition_download.return_value = str(tmp_path)
    with patch.dict("sys.modules", {"kagglehub": fake}):
        result = kaggle_competition_cached("ieee-fraud-detection")
    fake.competition_download.assert_called_once_with("ieee-fraud-detection")
    fake.dataset_download.assert_not_called()
    assert result == tmp_path


def test_dataset_download_is_used_for_datasets(fake_credentials, monkeypatch, tmp_path):
    monkeypatch.setenv("KAGGLEHUB_CACHE", str(tmp_path / "empty-cache"))
    fake = MagicMock()
    fake.dataset_download.return_value = str(tmp_path)
    with patch.dict("sys.modules", {"kagglehub": fake}):
        result = kaggle_dataset_cached("wordsforthewise/lending-club")
    fake.dataset_download.assert_called_once()
    assert result == tmp_path


def test_cached_dataset_is_reused_without_credentials_or_network(monkeypatch, tmp_path):
    cache_root = tmp_path / "kagglehub"
    cached_version = cache_root / "datasets" / "owner" / "slug" / "versions" / "3"
    cached_version.mkdir(parents=True)
    expected = cached_version / "dataset.csv"
    expected.write_text("label\n1\n")
    monkeypatch.setenv("KAGGLEHUB_CACHE", str(cache_root))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    fake = MagicMock()

    with patch.dict("sys.modules", {"kagglehub": fake}):
        found = kaggle_dataset_cached("owner/slug", filename="dataset.csv")

    assert found == expected
    fake.dataset_download.assert_not_called()


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


def _local_dataset_spec(*, expected_sha256: str | None, expected_rows: int) -> dict:
    return {
        "provenance": {
            "name": "Local test dataset",
            "kind": "kaggle_dataset",
            "slug": "owner/test",
            "url": "https://example.test/dataset",
            "licence": "test only",
            "access": "test only",
            "expected_sha256": expected_sha256,
            "expected_rows": expected_rows,
        },
        "approx_mb": 0,
        "expanded_mb": 0,
        "primary_file": "dataset.csv",
    }


def test_download_rejects_a_digest_that_does_not_match_provenance(monkeypatch, tmp_path):
    target = tmp_path / "dataset.csv"
    target.write_text("label,value\n0,10\n1,20\n")
    monkeypatch.setattr(
        download_data,
        "DATASETS",
        {"local": _local_dataset_spec(expected_sha256="0" * 64, expected_rows=2)},
    )
    monkeypatch.setattr(download_data, "kaggle_dataset_cached", lambda slug: tmp_path)

    with pytest.raises(DatasetAccessError) as excinfo:
        download_data._fetch("local")

    message = str(excinfo.value)
    assert "0" * 64 in message
    assert sha256_of(target) in message
    assert "delete" in message.lower()
    assert "retry" in message.lower()


def test_unpinned_digest_is_printed_with_exact_provenance_key_and_rows_are_checked(
    monkeypatch, tmp_path, capsys
):
    target = tmp_path / "dataset.csv"
    target.write_text("label,value\n0,10\n1,20\n")
    monkeypatch.setattr(
        download_data,
        "DATASETS",
        {"local": _local_dataset_spec(expected_sha256=None, expected_rows=3)},
    )
    monkeypatch.setattr(download_data, "kaggle_dataset_cached", lambda slug: tmp_path)

    record = download_data._fetch("local")

    output = capsys.readouterr().out
    assert "RECORD THIS" in output
    assert 'PROVENANCE["expected_sha256"]' in output
    assert record["rows"] == 2
    assert "ROW COUNT MISMATCH" in output
    assert "expected 3" in output
    assert "observed 2" in output


def test_openml_download_reports_shape_positive_rate_and_no_file_digest(monkeypatch, capsys):
    frame = pd.DataFrame(
        {
            "is_fraud": pd.Series([0, 0, 1], dtype="int8"),
            "Time": pd.Series([0, 1, 2], dtype="float32"),
        }
    )
    monkeypatch.setattr(download_data.ulb_creditcard, "load", lambda: frame)

    record = download_data._fetch("ulb-creditcard")

    assert record["rows"] == 3
    assert record["cols"] == 2
    assert record["positive_rate"] == pytest.approx(1 / 3)
    assert record["sha256"] == "n/a"
    output = capsys.readouterr().out
    assert "positive rate 0.333333" in output
    assert "sha256: n/a" in output


def test_openml_cache_restores_the_documented_row_id(tmp_path):
    details = {
        "row_id_attribute": "Time",
        "url": "https://openml.org/data/v1/download/42/tiny.arff",
    }
    cached = tmp_path / "openml" / "openml.org" / "data" / "v1" / "download" / "42"
    cached.mkdir(parents=True)
    with gzip.open(cached / "tiny.arff.gz", "wt") as handle:
        handle.write(
            "@relation tiny\n"
            "@attribute Time numeric\n"
            "@attribute V1 numeric\n"
            "@attribute Amount numeric\n"
            "@attribute Class {'0','1'}\n"
            "@data\n"
            "10,1.5,20,'0'\n"
            "20,2.5,30,'1'\n"
        )
    sklearn_frame = pd.DataFrame(
        {
            "V1": [1.5, 2.5],
            "Amount": [20.0, 30.0],
            "Class": ["0", "1"],
        }
    )

    restored = download._restore_openml_row_id(sklearn_frame, details, data_home=tmp_path)

    assert restored.columns[0] == "Time"
    assert restored["Time"].tolist() == [10.0, 20.0]


@pytest.mark.network
def test_real_kaggle_canary():
    """The one test that hits Kaggle. Deselected in CI with -m 'not network'."""
    path = kaggle_dataset_cached("sakshigoyal7/credit-card-customers")
    assert path.exists()


# ── credential sources ───────────────────────────────────────────────────────
#
# Three sources are supported and all three are exercised here, because the
# loader used to accept only two and refused a token that demonstrably works:
# with only ~/.kaggle/access_token present, kagglehub authenticates and downloads
# datasets, but ensure_kaggle_env() raised before kagglehub was ever called.


def test_env_vars_are_exported_for_kagglehub(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "deadbeef")

    from src.data.kaggle_credentials import ensure_kaggle_env

    ensure_kaggle_env()

    import os

    assert os.environ["KAGGLE_USERNAME"] == "tester"
    assert os.environ["KAGGLE_KEY"] == "deadbeef"


def test_kaggle_json_is_exported_for_kagglehub(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "kaggle.json").write_text('{"username": "from-file", "key": "0ddba11"}')

    from src.data.kaggle_credentials import ensure_kaggle_env

    ensure_kaggle_env()

    import os

    assert os.environ["KAGGLE_USERNAME"] == "from-file"
    assert os.environ["KAGGLE_KEY"] == "0ddba11"


def test_kagglehub_oauth_token_is_accepted_without_exporting_anything(monkeypatch, tmp_path):
    """`kagglehub login` writes access_token; kagglehub reads it itself."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "access_token").write_text("a-real-looking-oauth-token")

    import importlib

    from src.data import kaggle_credentials

    importlib.reload(kaggle_credentials)

    assert kaggle_credentials.has_kagglehub_oauth_token() is True
    kaggle_credentials.ensure_kaggle_env()  # must not raise

    import os

    assert "KAGGLE_USERNAME" not in os.environ

    importlib.reload(kaggle_credentials)


def test_an_empty_access_token_is_not_a_credential(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    kaggle_dir = tmp_path / ".kaggle"
    kaggle_dir.mkdir()
    (kaggle_dir / "access_token").write_text("")

    import importlib

    from src.data import kaggle_credentials

    importlib.reload(kaggle_credentials)
    assert kaggle_credentials.has_kagglehub_oauth_token() is False

    with pytest.raises(RuntimeError, match="kagglehub login"):
        kaggle_credentials.ensure_kaggle_env()

    importlib.reload(kaggle_credentials)
