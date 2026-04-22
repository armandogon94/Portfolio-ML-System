"""Configuration loader for YAML config files."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _backfill_defaults(config: dict) -> dict:
    """Populate non-destructive defaults on a loaded config.

    Phase A.1 introduces a three-modality data pipeline (synthetic | stream |
    mixed) driven by ``data.source``. Configs that predate A.1 (credit_risk,
    fraud_detection, demand_forecasting) never opt into modalities, so we
    backfill ``data.source = "synthetic"`` when the key is absent. This keeps
    downstream dispatchers (``train_price.py``, etc.) single-path without
    forcing every legacy config to be rewritten.

    The backfill is non-destructive: an explicitly set ``source`` (e.g. for
    ``price_prediction.yaml`` which sets ``stream`` or ``mixed``) is preserved.
    """
    data = config.get("data")
    if isinstance(data, dict) and "source" not in data:
        data["source"] = "synthetic"
    return config


def load_config(config_name: str) -> dict:
    """Load a YAML config file by problem name.

    Args:
        config_name: Name like 'credit_risk' or path to a YAML file.

    Returns:
        Parsed config dictionary with resolved paths and default modality
        source backfilled (see :func:`_backfill_defaults`).
    """
    if config_name.endswith(".yaml") or config_name.endswith(".yml"):
        config_path = Path(config_name)
    else:
        config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths against project root
    for key in ("raw_data_path", "processed_data_path", "checkpoint_dir", "results_dir"):
        if key in config.get("data", {}):
            config["data"][key] = str(PROJECT_ROOT / config["data"][key])
        if key in config.get("training", {}):
            config["training"][key] = str(PROJECT_ROOT / config["training"][key])

    # Phase A.1: ensure every config advertises a `data.source` modality.
    _backfill_defaults(config)

    return config


def get_project_root() -> Path:
    return PROJECT_ROOT
