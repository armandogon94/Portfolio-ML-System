"""Tests for src/data/kaggle_credentials.py — env + kaggle.json loader."""

from __future__ import annotations

import json
import os

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clear_kaggle_env(monkeypatch):
    """Ensure no Kaggle env vars leak in from the dev environment."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """Point HOME at a fresh tmp dir with no ~/.kaggle/."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _write_kaggle_json(home: os.PathLike, username: str, key: str) -> None:
    """Helper: drop a valid kaggle.json into ~/.kaggle/ under the given home."""
    from pathlib import Path

    kaggle_dir = Path(home) / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": username, "key": key})
    )


# ---------------------------------------------------------------------------
# load_kaggle_creds
# ---------------------------------------------------------------------------


class TestLoadKaggleCreds:

    def test_load_from_env_vars(self, clear_kaggle_env, isolated_home, monkeypatch):
        """Env vars alone are sufficient — no file required."""
        from src.data.kaggle_credentials import load_kaggle_creds

        monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
        monkeypatch.setenv("KAGGLE_KEY", "env_key_xyz")

        creds = load_kaggle_creds()

        assert creds == {"username": "env_user", "key": "env_key_xyz"}

    def test_load_from_kaggle_json(self, clear_kaggle_env, isolated_home):
        """Falls back to ~/.kaggle/kaggle.json when env vars absent."""
        from src.data.kaggle_credentials import load_kaggle_creds

        _write_kaggle_json(isolated_home, "file_user", "file_key_abc")

        creds = load_kaggle_creds()

        assert creds == {"username": "file_user", "key": "file_key_abc"}

    def test_env_wins_over_file(self, clear_kaggle_env, isolated_home, monkeypatch):
        """When both sources exist, env vars take precedence."""
        from src.data.kaggle_credentials import load_kaggle_creds

        _write_kaggle_json(isolated_home, "file_user", "file_key")
        monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
        monkeypatch.setenv("KAGGLE_KEY", "env_key")

        creds = load_kaggle_creds()

        assert creds == {"username": "env_user", "key": "env_key"}

    def test_raises_when_missing(self, clear_kaggle_env, isolated_home):
        """Raises RuntimeError with a message mentioning both setup options."""
        from src.data.kaggle_credentials import load_kaggle_creds

        with pytest.raises(RuntimeError) as excinfo:
            load_kaggle_creds()

        msg = str(excinfo.value)
        # Helpful message references both paths
        assert "KAGGLE_USERNAME" in msg
        assert "kaggle.json" in msg

    def test_partial_env_falls_back_to_file(
        self, clear_kaggle_env, isolated_home, monkeypatch
    ):
        """Only one env var set is treated as 'env unavailable' — file takes over."""
        from src.data.kaggle_credentials import load_kaggle_creds

        _write_kaggle_json(isolated_home, "file_user", "file_key")
        # Only username set, key missing
        monkeypatch.setenv("KAGGLE_USERNAME", "env_user")

        creds = load_kaggle_creds()

        assert creds == {"username": "file_user", "key": "file_key"}


# ---------------------------------------------------------------------------
# ensure_kaggle_env — side-effect helper for kagglehub
# ---------------------------------------------------------------------------


class TestEnsureKaggleEnv:

    def test_ensure_kaggle_env_exports_to_os_environ(
        self, clear_kaggle_env, isolated_home
    ):
        """ensure_kaggle_env() writes creds into os.environ for kagglehub."""
        from src.data.kaggle_credentials import ensure_kaggle_env

        _write_kaggle_json(isolated_home, "export_user", "export_key")

        ensure_kaggle_env()

        assert os.environ["KAGGLE_USERNAME"] == "export_user"
        assert os.environ["KAGGLE_KEY"] == "export_key"
