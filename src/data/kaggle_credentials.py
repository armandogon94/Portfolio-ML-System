"""Kaggle API credentials loader.

Three sources, in order of precedence:

1. Environment variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
2. ``~/.kaggle/kaggle.json``, the classic API token.
3. ``~/.kaggle/access_token``, the OAuth token written by ``kagglehub login``.

The third case exports nothing: kagglehub reads that file itself. This module
only has to stop refusing. That refusal was a real bug, measured on this
machine on 2026-07-25, ``kagglehub.auth.whoami()`` and
``kagglehub.dataset_download('sakshigoyal7/credit-card-customers')`` both
succeed with only ``access_token`` present, while this loader raised and the
download never got the chance.

One caveat, also measured rather than assumed: the OAuth token authenticates
Kaggle **datasets** but NOT **competitions**.
``kagglehub.competition_download('ieee-fraud-detection')`` returns
``403 ... have accepted the competition rules`` with it, even when the rules are
accepted, so IEEE-CIS still needs a classic ``kaggle.json``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MISSING_CREDS_MSG = (
    "Kaggle credentials not found. Provide them in any one of three ways:\n"
    "  (1) environment variables: export KAGGLE_USERNAME=... KAGGLE_KEY=...\n"
    "  (2) a kaggle.json file at ~/.kaggle/kaggle.json\n"
    "  (3) run `kagglehub login`, which writes ~/.kaggle/access_token\n"
    "Get an API key at https://www.kaggle.com/settings/account -> 'Create New Token'.\n"
    "Note: option (3) authenticates datasets but NOT competitions, so IEEE-CIS\n"
    "needs option (1) or (2)."
)


def kagglehub_token_path() -> Path:
    """Path of the token written by ``kagglehub login``.

    Resolved on every call rather than bound at import. As a module-level
    constant this was evaluated once, against whatever ``Path.home()`` returned
    the first time the module happened to be imported, so a caller that
    redirects the home directory (the test suite does, to assert the
    no-credentials path) permanently pinned the constant to a temporary
    directory, and every later credential check in the same process reported
    "no token" against a home that was never real. ``load_kaggle_creds`` already
    resolves ``Path.home()`` per call; this now matches it.
    """
    return Path.home() / ".kaggle" / "access_token"


def has_kagglehub_oauth_token() -> bool:
    """True when ``kagglehub login`` has left a token kagglehub can use itself."""
    try:
        token = kagglehub_token_path()
        return token.is_file() and token.stat().st_size > 0
    except OSError:
        return False


def load_kaggle_creds() -> dict[str, str]:
    """Return Kaggle credentials as ``{"username": ..., "key": ...}``.

    Environment variables take precedence. Falls back to
    ``~/.kaggle/kaggle.json`` if either env var is missing.

    Raises:
        RuntimeError: If no valid credentials can be found.
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if username and key:
        logger.debug("Loaded Kaggle credentials from environment variables")
        return {"username": username, "key": key}

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        data = json.loads(kaggle_json.read_text())
        file_username = data.get("username")
        file_key = data.get("key")
        if file_username and file_key:
            logger.debug("Loaded Kaggle credentials from %s", kaggle_json)
            return {"username": file_username, "key": file_key}

    raise RuntimeError(_MISSING_CREDS_MSG)


def ensure_kaggle_env() -> None:
    """Make sure kagglehub will find *some* credential, or raise with instructions.

    Env vars and ``kaggle.json`` are exported into ``os.environ`` because that is
    how kagglehub discovers them. An ``access_token`` needs no export (kagglehub
    reads it itself), so this function simply returns without raising.

    Raises:
        RuntimeError: No credential of any of the three kinds is available.
    """
    try:
        creds = load_kaggle_creds()
    except RuntimeError:
        if has_kagglehub_oauth_token():
            logger.debug("Using the kagglehub OAuth token at %s", kagglehub_token_path())
            return
        raise
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
