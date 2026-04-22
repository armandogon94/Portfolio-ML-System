"""Kaggle API credentials loader.

Reads credentials from (in order of precedence):
1. Environment variables KAGGLE_USERNAME and KAGGLE_KEY
2. File at ~/.kaggle/kaggle.json (standard Kaggle location)

Used before any kagglehub operation so that the kagglehub library
picks up valid credentials via its own env-var lookup.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MISSING_CREDS_MSG = (
    "Kaggle credentials not found. Provide them via either:\n"
    "  (1) environment variables: export KAGGLE_USERNAME=... KAGGLE_KEY=...\n"
    "  (2) a kaggle.json file at ~/.kaggle/kaggle.json\n"
    "Get your API key from https://www.kaggle.com/settings/account → 'Create New Token'."
)


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
    """Export credentials into ``os.environ`` so kagglehub can pick them up.

    Called before ``kagglehub.dataset_download()`` in ``src/data/stream.py``.
    Raises RuntimeError if credentials are unavailable.
    """
    creds = load_kaggle_creds()
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
