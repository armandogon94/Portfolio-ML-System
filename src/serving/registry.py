"""Checkpoint discovery and lazy loading.

Replaces the discovery half of the 573-line ``predictor.py`` monolith. Models are
found by globbing ``checkpoints/*/metadata.json``, so adding a model requires no
change to this file and no change to ``api.py``.

Loading is lazy and cached. A cold ``/predict/fraud`` pays the joblib deserialise
once; every later request is a dict lookup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_project_root

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A deserialised checkpoint plus everything needed to score with it."""

    problem: str
    model: Any
    feature_columns: list[str]
    feature_artifacts: dict[str, Any] = field(default_factory=dict)
    #: Training category sets, replayed on every request. See src/features/schema.py.
    category_dtypes: dict[str, list] = field(default_factory=dict)
    preprocessor: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointRegistry:
    """Discovers, loads and caches model checkpoints.

    Args:
        root: Directory holding ``<problem>/metadata.json`` subdirectories.
            Defaults to ``<repo>/checkpoints``. Overridable so tests can point at
            a tmp_path without monkeypatching.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else get_project_root() / "checkpoints"
        self._cache: dict[str, LoadedModel] = {}

    # ── discovery ────────────────────────────────────────────────────────────

    def available(self) -> list[str]:
        """Problem names with a readable checkpoint on disk, sorted."""
        if not self.root.is_dir():
            return []
        return sorted(
            path.parent.name for path in self.root.glob("*/metadata.json") if _readable(path)
        )

    def model_info(self) -> dict[str, dict[str, Any]]:
        """Return every checkpoint's ``metadata.json``, keyed by problem.

        Corrupt or partially-written metadata is skipped rather than crashing the
        ``/models`` endpoint: a half-flushed checkpoint during a training run
        should not take the API down.
        """
        info: dict[str, dict[str, Any]] = {}
        if not self.root.is_dir():
            return info
        for path in sorted(self.root.glob("*/metadata.json")):
            try:
                info[path.parent.name] = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable checkpoint %s: %s", path, exc)
        return info

    # ── loading ──────────────────────────────────────────────────────────────

    def load(self, problem: str) -> LoadedModel:
        """Return the loaded model for ``problem``, deserialising on first use.

        Raises:
            FileNotFoundError: No checkpoint exists, with the exact command that
                would create one. A bare "file not found" here sends people
                hunting for a path bug when the real answer is "train it first".
        """
        if problem in self._cache:
            return self._cache[problem]

        directory = self.root / problem
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No checkpoint for {problem!r} at {directory}.\n"
                f"Available: {self.available() or 'none'}\n"
                f"Create it with:\n"
                f"  uv run python scripts/download_data.py --dataset all\n"
                f"  uv run python scripts/train.py --model {problem}"
            )

        import joblib

        metadata = json.loads(metadata_path.read_text())
        bundle = joblib.load(directory / "features.joblib")
        if metadata.get("model_type") == "autoencoder":
            import torch

            from src.models.autoencoder import FraudAutoencoder

            payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            model = FraudAutoencoder(
                input_dim=int(payload["input_dim"]),
                hidden_dims=list(payload["hidden_dims"]),
                dropout=float(payload.get("dropout", 0.1)),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
        else:
            model = joblib.load(directory / "model.joblib")
        loaded = LoadedModel(
            problem=problem,
            model=model,
            feature_columns=list(bundle["feature_columns"]),
            feature_artifacts=dict(bundle.get("artifacts", {})),
            category_dtypes=dict(bundle.get("category_dtypes", {})),
            preprocessor=bundle.get("preprocessor"),
            metadata=metadata,
        )
        self._cache[problem] = loaded
        logger.info(
            "Loaded %s: %s, %d features, git %s",
            problem,
            metadata.get("model_type"),
            len(loaded.feature_columns),
            str(metadata.get("git_sha"))[:8],
        )
        return loaded

    def invalidate(self, problem: str | None = None) -> None:
        """Drop cached models so a retrained checkpoint is picked up."""
        if problem is None:
            self._cache.clear()
        else:
            self._cache.pop(problem, None)


def _readable(path: Path) -> bool:
    try:
        json.loads(path.read_text())
        return True
    except (json.JSONDecodeError, OSError):
        return False
