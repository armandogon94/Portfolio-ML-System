"""Tests for config loading and project root."""

from pathlib import Path

import pytest

from src.config import get_project_root, load_config


class TestGetProjectRoot:

    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_configs(self):
        root = get_project_root()
        assert (root / "configs").is_dir()


class TestLoadConfig:

    def test_load_by_name(self):
        config = load_config("credit_risk")
        assert config["problem"] == "credit_risk"
        assert "data" in config
        assert "model" in config

    def test_load_by_yaml_path(self):
        root = get_project_root()
        config = load_config(str(root / "configs" / "credit_risk.yaml"))
        assert config["problem"] == "credit_risk"

    def test_invalid_config_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config")

    def test_paths_resolved(self):
        config = load_config("credit_risk")
        raw_path = config["data"]["raw_data_path"]
        assert Path(raw_path).is_absolute()

    @pytest.mark.parametrize("config_name", [
        "credit_risk", "fraud_detection", "price_prediction", "demand_forecasting",
    ])
    def test_all_configs_loadable(self, config_name):
        config = load_config(config_name)
        assert "problem" in config
