"""Configuration loader and validator for three fintech problems.

Config filenames identify dataset-specific runs, so the fraud problem has both
``fraud.yaml`` (IEEE-CIS) and ``fraud_ulb.yaml`` (OpenML 1597). A config is the
whole run specification: source, split, features, model, metrics, and sanity band.

Validation happens at load time. A typo in a config fails immediately rather
than at fold 3 of a long run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Problem names with a config in ``configs/``. Deliberately closed —
#: the repo is three fintech problems, not an open catalogue.
PROBLEMS = ("fraud", "credit_risk", "churn")

_REQUIRED_TOP_LEVEL = ("problem", "seed", "data", "split", "features", "model")
_PATH_KEYS = ("sample_path", "processed_path")
_SPLIT_TYPES = ("time", "stratified_kfold", "random")
_SOURCE_KINDS = ("kaggle_competition", "kaggle_dataset", "openml")

# ADR-0003: names associated with generated or simulated data are forbidden in
# both source kinds and adapter paths so synthetic labels cannot re-enter a run.
FORBIDDEN_SOURCE_MARKERS = (
    "synthetic",
    "simulated",
    "simulate",
    "generate",
    "generator",
    "make_",
    "fake",
    "mock",
)


class ConfigError(ValueError):
    """Raised when a config file is missing a required key or is self-inconsistent."""


def config_path(config_name: str) -> Path:
    """Resolve a problem name or explicit path to a config file path."""
    if config_name.endswith((".yaml", ".yml")):
        return Path(config_name)
    return PROJECT_ROOT / "configs" / f"{config_name}.yaml"


def available_config_names() -> tuple[str, ...]:
    """Return training config filenames independently of the closed problem set."""
    return tuple(sorted(path.stem for path in (PROJECT_ROOT / "configs").glob("*.yaml")))


def load_config(config_name: str) -> dict[str, Any]:
    """Load, resolve and validate a problem config.

    Args:
        config_name: A problem name (``"fraud"``) or a path to a YAML file.

    Returns:
        The parsed config with relative paths resolved against the project root.

    Raises:
        FileNotFoundError: The config file does not exist.
        ConfigError: The config is missing a required key or is inconsistent.
    """
    path = config_path(config_name)
    if not path.exists():
        available = list(available_config_names())
        raise FileNotFoundError(f"Config not found: {path}. Available: {available}")

    with open(path) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError(f"{path} did not parse to a mapping.")

    _validate(config, path)
    _resolve_paths(config)
    return config


def _validate(config: dict[str, Any], path: Path) -> None:
    """Fail loudly and specifically on a malformed config."""
    missing = [key for key in _REQUIRED_TOP_LEVEL if key not in config]
    if missing:
        raise ConfigError(f"{path}: missing required top-level keys: {missing}")

    data = config["data"]
    if "source" not in data:
        raise ConfigError(
            f"{path}: data.source is required. Every config must name a real, "
            f"downloadable dataset — synthetic generators were removed in ADR-0003."
        )
    source = data["source"]
    if not isinstance(source, dict):
        raise ConfigError(
            f"{path}: data.source must be a mapping. Add kind, dataset identity, "
            "and an adapter under data.source."
        )
    if not source.get("kind") or not source.get("adapter"):
        raise ConfigError(f"{path}: data.source needs both 'kind' and 'adapter'.")

    kind = source["kind"]
    adapter = source["adapter"]
    if not isinstance(kind, str) or not isinstance(adapter, str):
        raise ConfigError(
            f"{path}: data.source.kind and data.source.adapter must be strings. "
            "Name a supported download source and its adapter module."
        )

    for field, value in (("kind", kind), ("adapter", adapter)):
        lowered = value.casefold()
        marker = next((term for term in FORBIDDEN_SOURCE_MARKERS if term in lowered), None)
        if marker is not None:
            raise ConfigError(
                f"{path}: data.source.{field}={value!r} is forbidden by ADR-0003 "
                f"because it contains {marker!r}. Point this config at a real "
                "Kaggle or OpenML dataset and a canonical src.data.adapters module."
            )

    if kind not in _SOURCE_KINDS:
        raise ConfigError(
            f"{path}: data.source.kind {kind!r} is unsupported. Use one of "
            f"{_SOURCE_KINDS} and name the dataset's real download identifier."
        )

    if kind in {"kaggle_competition", "kaggle_dataset"}:
        slug = source.get("slug")
        if not isinstance(slug, str) or not slug.strip():
            raise ConfigError(
                f"{path}: data.source.slug is required for {kind!r}. Add the "
                "Kaggle competition slug or owner/dataset slug shown on its source page."
            )
    elif not isinstance(source.get("data_id"), int) or isinstance(source.get("data_id"), bool):
        raise ConfigError(
            f"{path}: data.source.data_id is required for 'openml' and must be an "
            "integer. Add the numeric OpenML dataset id from its source page."
        )

    if not adapter.startswith("src.data.adapters."):
        raise ConfigError(
            f"{path}: data.source.adapter {adapter!r} must start with "
            "'src.data.adapters.'. Move the canonical adapter there or correct "
            "the dotted module path."
        )
    try:
        adapter_spec = importlib.util.find_spec(adapter)
    except (ImportError, ModuleNotFoundError, ValueError):
        adapter_spec = None
    if adapter_spec is None:
        raise ConfigError(
            f"{path}: data.source.adapter {adapter!r} does not resolve to a module. "
            "Add that adapter module or correct the dotted path before training."
        )

    if "target" not in data:
        raise ConfigError(f"{path}: data.target is required.")

    split_type = config["split"].get("type")
    if split_type not in _SPLIT_TYPES:
        raise ConfigError(
            f"{path}: unknown split.type {split_type!r}. Expected one of {_SPLIT_TYPES}."
        )
    if split_type == "time" and not config["split"].get("column"):
        raise ConfigError(f"{path}: split.type 'time' requires split.column.")

    if not config["model"].get("type"):
        raise ConfigError(f"{path}: model.type is required.")

    band = config.get("sanity_band")
    if not isinstance(band, dict):
        raise ConfigError(
            f"{path}: sanity_band is required and must be a mapping with "
            "metric, min, and an optional max. It is an expected-range smoke "
            "alarm, not a leakage test."
        )
    if band.get("metric") not in {"pr_auc", "roc_auc"}:
        raise ConfigError(f"{path}: sanity_band.metric must be 'pr_auc' or 'roc_auc'.")
    if not isinstance(band.get("min"), (int, float)):
        raise ConfigError(f"{path}: sanity_band.min must be numeric.")
    if "max" in band and not isinstance(band["max"], (int, float)):
        raise ConfigError(f"{path}: sanity_band.max must be numeric when present.")
    if "max" in band and band["min"] >= band["max"]:
        raise ConfigError(f"{path}: sanity_band.min must be less than sanity_band.max.")

    # A denylist that does not contain the target is a footgun: the target column
    # would otherwise be eligible for selection as a feature.
    denylist = data.get("denylist")
    if denylist is not None and data["target"] not in denylist:
        raise ConfigError(
            f"{path}: data.target {data['target']!r} must appear in data.denylist "
            f"so it can never be selected as a feature."
        )


def _resolve_paths(config: dict[str, Any]) -> None:
    """Make every path in the config absolute, relative to the project root."""
    for section, keys in (
        ("data", _PATH_KEYS),
        ("training", ("checkpoint_dir", "reports_dir")),
    ):
        block = config.get(section, {})
        for key in keys:
            if key in block:
                value = Path(block[key])
                block[key] = str(value if value.is_absolute() else PROJECT_ROOT / value)


def get_project_root() -> Path:
    """Return the repository root as a ``Path``."""
    return PROJECT_ROOT
