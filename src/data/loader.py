"""
loader.py

Purpose: Dataset loading utilities. This module ONLY loads and reports on
raw CSVs — it must never perform model-specific transformations (cleaning,
encoding, scaling belong in src/preprocessing/).
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd

# pyrefly: ignore [missing-import]
from src.utils.config import load_config
# pyrefly: ignore [missing-import]
from src.utils.paths import get_raw_data_dir
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Read a single CSV file safely, with existence validation and a
    shape/columns/dtypes/missing-values report."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Check that the file exists at this exact path."
        )

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    _report(df, path.name)
    return df


def _report(df: pd.DataFrame, label: str) -> None:
    logger.info(f"[{label}] shape: {df.shape[0]} rows x {df.shape[1]} columns")
    logger.info(f"[{label}] columns: {list(df.columns)}")
    logger.info(f"[{label}] dtypes:\n{df.dtypes.to_string()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info(f"[{label}] missing values: none")
    else:
        logger.info(f"[{label}] missing values:\n{missing.to_string()}")


def load_training_data(config: Optional[dict] = None) -> pd.DataFrame:
    """Load data.training_file from paths.raw_data, as configured in config.yaml."""
    config = config or load_config()
    raw_dir = get_raw_data_dir(config)
    file_path = raw_dir / config["data"]["training_file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {file_path}\n"
            f"Download UNSW-NB15 and place '{config['data']['training_file']}' "
            f"in '{raw_dir}'. See README.md -> 'Dataset Setup'."
        )
    return load_dataset(file_path)


def load_testing_data(config: Optional[dict] = None) -> pd.DataFrame:
    """Load data.testing_file from paths.raw_data, as configured in config.yaml."""
    config = config or load_config()
    raw_dir = get_raw_data_dir(config)
    file_path = raw_dir / config["data"]["testing_file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Testing file not found: {file_path}\n"
            f"Download UNSW-NB15 and place '{config['data']['testing_file']}' "
            f"in '{raw_dir}'. See README.md -> 'Dataset Setup'."
        )
    return load_dataset(file_path)
