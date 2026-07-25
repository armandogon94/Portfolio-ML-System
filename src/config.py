"""Configuration loader and validator for the three fintech problem configs.

One config per problem in ``configs/``. A config is the whole specification of a
training run: where the data comes from, how it is split, which model is built,
which metrics are reported, and the band outside which the result is a bug.

Validation happens at load time. A typo in a config fails immediately rather
than at fold 3 of a long run.
"""

from __future__ import annotations

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


class ConfigError(ValueError):
    """Raised when a config file is missing a required key or is self-inconsistent."""


def config_path(config_name: str) -> Path:
    """Resolve a problem name or explicit path to a config file path."""
    if config_name.endswith((".yaml", ".yml")):
        return Path(config_name)
    return PROJECT_ROOT / "configs" / f"{config_name}.yaml"


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
        available = sorted(p.stem for p in (PROJECT_ROOT / "configs").glob("*.yaml"))
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
    if not source.get("kind") or not source.get("adapter"):
        raise ConfigError(f"{path}: data.source needs both 'kind' and 'adapter'.")
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
