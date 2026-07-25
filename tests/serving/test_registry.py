"""Checkpoint discovery and lazy loading."""

from __future__ import annotations

import json

import pytest

from src.serving.registry import CheckpointRegistry


def test_discovers_the_checkpoint_on_disk(registry):
    assert registry.available() == ["churn"]


def test_model_info_returns_the_full_metadata(registry):
    info = registry.model_info()
    assert "churn" in info
    assert info["churn"]["model_type"] == "lightgbm"
    assert "git_sha" in info["churn"]


def test_load_returns_the_model_and_its_feature_columns(registry):
    loaded = registry.load("churn")
    assert loaded.problem == "churn"
    assert hasattr(loaded.model, "predict_proba")
    assert loaded.feature_columns


def test_load_is_cached(registry):
    """A cold request pays the deserialise once; later ones are a dict lookup."""
    assert registry.load("churn") is registry.load("churn")


def test_invalidate_forces_a_reload(registry):
    first = registry.load("churn")
    registry.invalidate("churn")
    assert registry.load("churn") is not first


def test_a_missing_checkpoint_names_the_command_that_creates_it(registry):
    """A bare FileNotFoundError sends people hunting for a path bug."""
    with pytest.raises(FileNotFoundError) as excinfo:
        registry.load("fraud")
    message = str(excinfo.value)
    assert "scripts/train.py --model fraud" in message
    assert "download_data.py" in message


def test_an_empty_root_is_not_an_error(tmp_path):
    """A fresh clone has no checkpoints. /models must still answer."""
    empty = CheckpointRegistry(tmp_path / "nope")
    assert empty.available() == []
    assert empty.model_info() == {}


def test_corrupt_metadata_is_skipped_not_fatal(tmp_path):
    """A half-flushed checkpoint during training must not take the API down."""
    root = tmp_path / "checkpoints"
    (root / "broken").mkdir(parents=True)
    (root / "broken" / "metadata.json").write_text("{ not json")
    (root / "good").mkdir()
    (root / "good" / "metadata.json").write_text(json.dumps({"problem": "good"}))

    reg = CheckpointRegistry(root)
    assert reg.available() == ["good"]
    assert list(reg.model_info()) == ["good"]
