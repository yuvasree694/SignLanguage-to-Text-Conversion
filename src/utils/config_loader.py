"""Configuration loader."""

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration and populate derived fields."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    alphabet = config["classes"]["alphabet"]
    words = config["classes"]["words"]
    mode = config["classes"].get("mode", "full")

    if mode == "alphabet_only":
        config["classes"]["all"] = list(alphabet)
    else:
        config["classes"]["all"] = alphabet + [w.upper() for w in words]
    config["num_classes"] = len(config["classes"]["all"])

    return config


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """Resolve path relative to project root."""
    return get_project_root() / relative_path


def get_class_labels(config: Dict[str, Any]) -> List[str]:
    """Return ordered class label list."""
    return config["classes"]["all"]
