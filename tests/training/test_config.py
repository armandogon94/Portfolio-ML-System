"""Config loading and validation. A typo must fail at load, not at fold 3."""

from __future__ import annotations

import pytest
import yaml

from src.config import PROBLEMS, ConfigError, load_config

REAL_SOURCE_KINDS = {"kaggle_competition", "kaggle_dataset", "openml"}
CONFIG_PROBLEMS = {
    "fraud": "fraud",
    "fraud_ulb": "fraud",
    "credit_risk": "credit_risk",
    "churn": "churn",
}


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_problem_config_loads(problem):
    config = load_config(problem)
    assert config["problem"] == problem
    assert isinstance(config["seed"], int)


@pytest.mark.parametrize(("config_name", "problem"), CONFIG_PROBLEMS.items())
def test_every_training_config_loads(config_name, problem):
    config = load_config(config_name)
    assert config["problem"] == problem
    assert config["seed"] == 42


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_config_names_a_real_downloadable_dataset(problem):
    """No config may point at a generator. That is the whole rebuild in one test."""
    source = load_config(problem)["data"]["source"]
    assert source["kind"] in REAL_SOURCE_KINDS
    assert source["adapter"].startswith("src.data.adapters.")
    assert "generate" not in source["adapter"]


@pytest.mark.parametrize("config_name", CONFIG_PROBLEMS)
def test_every_config_declares_a_seed_and_any_sanity_band_is_bounded(config_name):
    config = load_config(config_name)
    assert config["seed"] == 42
    assert "expected" not in config
    if "sanity_band" in config:
        assert config["sanity_band"]["metric"] in {"pr_auc", "roc_auc"}
        assert "min" in config["sanity_band"]


def test_config_files_cover_three_problems_and_two_real_fraud_datasets(project_root):
    """The business scope stays at three problems; fraud has a credential-free path."""
    configs = sorted(p.stem for p in (project_root / "configs").glob("*.yaml"))
    assert configs == ["churn", "credit_risk", "fraud", "fraud_ulb"]
    assert PROBLEMS == ("fraud", "credit_risk", "churn")


def test_ulb_has_no_posthoc_temporal_sanity_range():
    assert "sanity_band" not in load_config("fraud_ulb")


def test_churn_has_no_vacuous_upper_bound():
    assert "max" not in load_config("churn")["sanity_band"]


def test_missing_config_lists_what_is_available():
    with pytest.raises(FileNotFoundError, match="Available"):
        load_config("heart_disease")


def _write(tmp_path, config: dict):
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _minimal_config(source: dict) -> dict:
    return {
        "problem": "fraud",
        "seed": 42,
        "data": {"target": "is_fraud", "source": source},
        "split": {"type": "random"},
        "features": {"module": "src.features.fraud_features"},
        "model": {"type": "lightgbm"},
        "sanity_band": {"metric": "roc_auc", "min": 0.5},
    }


@pytest.mark.parametrize(
    "source",
    [
        {
            "kind": "synthetic_generator",
            "adapter": "src.data.generate_fraud",
        },
        {
            "kind": "openml",
            "data_id": 1597,
            "adapter": "src.synthetic.make_fraud",
        },
    ],
)
def test_synthetic_or_generator_sources_are_refused(tmp_path, source):
    path = _write(tmp_path, _minimal_config(source))
    with pytest.raises(ConfigError, match="ADR-0003"):
        load_config(path)


def test_kaggle_competition_without_slug_is_refused(tmp_path):
    path = _write(
        tmp_path,
        _minimal_config(
            {
                "kind": "kaggle_competition",
                "adapter": "src.data.adapters.ieee_cis",
            }
        ),
    )
    with pytest.raises(ConfigError, match=r"data\.source\.slug"):
        load_config(path)


def test_openml_without_data_id_is_refused(tmp_path):
    path = _write(
        tmp_path,
        _minimal_config(
            {
                "kind": "openml",
                "adapter": "src.data.adapters.ieee_cis",
            }
        ),
    )
    with pytest.raises(ConfigError, match=r"data\.source\.data_id"):
        load_config(path)


def test_adapter_must_resolve_to_a_real_module(tmp_path):
    path = _write(
        tmp_path,
        _minimal_config(
            {
                "kind": "openml",
                "data_id": 1597,
                "adapter": "src.data.adapters.does_not_exist",
            }
        ),
    )
    with pytest.raises(ConfigError, match="does not resolve"):
        load_config(path)


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
    config = _minimal_config(
        {
            "kind": "openml",
            "data_id": 1597,
            "adapter": "src.data.adapters.ieee_cis",
        }
    )
    config["split"] = {"type": "time"}
    path = _write(tmp_path, config)
    with pytest.raises(ConfigError, match="requires split.column"):
        load_config(path)


def test_a_denylist_that_omits_the_target_is_refused(tmp_path):
    """Otherwise the target is eligible for selection as a feature."""
    config = _minimal_config(
        {
            "kind": "openml",
            "data_id": 1597,
            "adapter": "src.data.adapters.ieee_cis",
        }
    )
    config["data"]["denylist"] = ["something_else"]
    path = _write(tmp_path, config)
    with pytest.raises(ConfigError, match="must appear in data.denylist"):
        load_config(path)


def test_an_unknown_split_type_is_refused(tmp_path):
    config = _minimal_config(
        {
            "kind": "openml",
            "data_id": 1597,
            "adapter": "src.data.adapters.ieee_cis",
        }
    )
    config["split"] = {"type": "bootstrap"}
    path = _write(tmp_path, config)
    with pytest.raises(ConfigError, match="unknown split.type"):
        load_config(path)
