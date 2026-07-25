"""Config loading and validation. A typo must fail at load, not at fold 3."""

from __future__ import annotations

import pytest
import yaml

from src.config import PROBLEMS, ConfigError, load_config


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_problem_config_loads(problem):
    config = load_config(problem)
    assert config["problem"] == problem
    assert isinstance(config["seed"], int)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_config_names_a_real_downloadable_dataset(problem):
    """No config may point at a generator. That is the whole rebuild in one test."""
    source = load_config(problem)["data"]["source"]
    assert source["kind"] in {"kaggle_competition", "kaggle_dataset", "openml"}
    assert source["adapter"].startswith("src.data.adapters.")
    assert "generate" not in source["adapter"]


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_config_declares_a_seed_and_an_expected_band(problem):
    config = load_config(problem)
    assert config["seed"] == 42
    assert "roc_auc_min" in config["expected"]
    assert "roc_auc_max" in config["expected"]


def test_there_are_exactly_three_configs(project_root):
    """Ten unrelated industries was the tell. Three fintech problems is the point."""
    configs = sorted(p.stem for p in (project_root / "configs").glob("*.yaml"))
    assert configs == ["churn", "credit_risk", "fraud"]


def test_missing_config_lists_what_is_available():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_config("heart_disease")


def _write(tmp_path, config: dict):
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_a_config_without_a_data_source_is_refused(tmp_path):
    path = _write(
        tmp_path,
        {
            "problem": "x",
            "seed": 1,
            "data": {"target": "y"},
            "split": {"type": "random"},
            "features": {"module": "m"},
            "model": {"type": "lightgbm"},
        },
    )
    with pytest.raises(ConfigError, match="data.source is required"):
        load_config(path)


def test_a_time_split_without_a_column_is_refused(tmp_path):
    path = _write(
        tmp_path,
        {
            "problem": "x",
            "seed": 1,
            "data": {"target": "y", "source": {"kind": "openml", "adapter": "a"}},
            "split": {"type": "time"},
            "features": {"module": "m"},
            "model": {"type": "lightgbm"},
        },
    )
    with pytest.raises(ConfigError, match="requires split.column"):
        load_config(path)


def test_a_denylist_that_omits_the_target_is_refused(tmp_path):
    """Otherwise the target is eligible for selection as a feature."""
    path = _write(
        tmp_path,
        {
            "problem": "x",
            "seed": 1,
            "data": {
                "target": "y",
                "source": {"kind": "openml", "adapter": "a"},
                "denylist": ["something_else"],
            },
            "split": {"type": "random"},
            "features": {"module": "m"},
            "model": {"type": "lightgbm"},
        },
    )
    with pytest.raises(ConfigError, match="must appear in data.denylist"):
        load_config(path)


def test_an_unknown_split_type_is_refused(tmp_path):
    path = _write(
        tmp_path,
        {
            "problem": "x",
            "seed": 1,
            "data": {"target": "y", "source": {"kind": "openml", "adapter": "a"}},
            "split": {"type": "bootstrap"},
            "features": {"module": "m"},
            "model": {"type": "lightgbm"},
        },
    )
    with pytest.raises(ConfigError, match="unknown split.type"):
        load_config(path)
