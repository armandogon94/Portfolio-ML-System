"""Configuration loader for YAML config files."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_name: str) -> dict:
    """Load a YAML config file by problem name.

    Args:
        config_name: Name like 'credit_risk' or path to a YAML file.

    Returns:
        Parsed config dictionary with resolved paths.
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

    return config


def get_project_root() -> Path:
    return PROJECT_ROOT
