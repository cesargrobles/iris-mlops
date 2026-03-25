# src/config.py
"""Configuration loader and validator for the MLOps pipeline."""

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load YAML configuration file and return as dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    return config
