"""
paths.py

Purpose: Resolve project paths from config.yaml. No other file in this
project should hardcode a filesystem path — everything goes through here
or through config/config.yaml.
"""
from pathlib import Path


def get_project_root() -> Path:
    """src/utils/paths.py -> src/utils -> src -> <project root>."""
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return get_project_root() / relative_path


def get_raw_data_dir(config: dict) -> Path:
    return resolve_path(config["paths"]["raw_data"])


def get_interim_data_dir(config: dict) -> Path:
    return resolve_path(config["paths"]["interim_data"])


def get_processed_data_dir(config: dict) -> Path:
    return resolve_path(config["paths"]["processed_data"])


def get_models_dir(config: dict) -> Path:
    return resolve_path(config["paths"]["models"])


def get_results_dir(config: dict) -> Path:
    return resolve_path(config["paths"]["results"])
