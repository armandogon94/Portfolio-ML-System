"""Tests for config loader modality backfill + extended price_prediction schema.

Covers Phase A.1 changes:
- `configs/price_prediction.yaml` gains `data.source`, `data.kaggle_slug`,
  `data.stream_file`, `data.adapter` keys for three-modality dispatch
  (synthetic | stream | mixed).
- `src/config.py::load_config` backfills `data.source = "synthetic"` on any
  config whose `data` block lacks the key, preserving backward compatibility
  for the three existing configs (credit_risk, fraud_detection,
  demand_forecasting).
"""

import pytest
import yaml

from src.config import load_config


class TestPricePredictionSchema:
    """The extended price_prediction.yaml should surface new streaming keys."""

    def test_price_config_has_source_field(self):
        config = load_config("price_prediction")
        assert config["data"]["source"] == "synthetic"

    def test_price_config_has_kaggle_slug(self):
        config = load_config("price_prediction")
        assert (
            config["data"]["kaggle_slug"]
            == "computingvictor/zillow-market-analysis-and-real-estate-sales-data"
        )
        assert config["data"]["adapter"] == "housing_adapter"

    def test_price_config_has_stream_file(self):
        config = load_config("price_prediction")
        assert "stream_file" in config["data"]


class TestDataSourceBackfill:
    """`load_config` must backfill data.source = 'synthetic' when missing."""

    def test_missing_source_defaults_to_synthetic(self, tmp_path):
        yaml_path = tmp_path / "tmp_no_source.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "problem": "tmp_no_source",
                    "data": {
                        "raw_data_path": "data/raw/tmp.csv",
                        "n_samples": 10,
                    },
                    "model": {"type": "none"},
                }
            )
        )

        config = load_config(str(yaml_path))

        assert config["data"]["source"] == "synthetic"

    def test_explicit_source_not_overwritten(self, tmp_path):
        yaml_path = tmp_path / "tmp_explicit_source.yaml"
        yaml_path.write_text(
            yaml.safe_dump(
                {
                    "problem": "tmp_explicit",
                    "data": {
                        "raw_data_path": "data/raw/tmp.csv",
                        "source": "stream",
                    },
                    "model": {"type": "none"},
                }
            )
        )

        config = load_config(str(yaml_path))

        assert config["data"]["source"] == "stream"

    @pytest.mark.parametrize(
        "config_name",
        ["credit_risk", "fraud_detection", "price_prediction", "demand_forecasting"],
    )
    def test_existing_configs_backward_compat(self, config_name):
        """All 4 real configs load and carry data.source == 'synthetic'."""
        config = load_config(config_name)
        assert "data" in config
        assert config["data"]["source"] == "synthetic"
