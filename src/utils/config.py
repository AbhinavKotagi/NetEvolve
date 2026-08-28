"""
config.py

Purpose: Load and validate config.yaml and classes.yaml. This is the only
module that should call yaml.safe_load directly.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from utils.paths import get_project_root


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    return data


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config/config.yaml (or an explicit override path)."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    return load_yaml(config_path)


def load_classes(classes_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load config/classes.yaml and validate it: both keys must exist, be
    non-empty lists, and must not overlap.
    """
    if classes_path is None:
        classes_path = get_project_root() / "config" / "classes.yaml"
    classes = load_yaml(classes_path)

    for key in ("known_classes", "future_unknown_classes"):
        if key not in classes:
            raise KeyError(f"'{key}' missing from classes.yaml")
        if not isinstance(classes[key], list) or len(classes[key]) == 0:
            raise ValueError(f"'{key}' in classes.yaml must be a non-empty list")

    overlap = set(classes["known_classes"]) & set(classes["future_unknown_classes"])
    if overlap:
        raise ValueError(f"Classes cannot be both known and future-unknown: {sorted(overlap)}")

    return classes
