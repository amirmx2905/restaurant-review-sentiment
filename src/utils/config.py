from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


ConfigDict = Dict[str, Any]


def load_config(config_path: str) -> ConfigDict:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping at the top level")

    return config
