"""
run_preprocessing.py

CLI entry point:
  Load raw data -> Clean -> Filter known classes -> Split -> Fit+transform ->
  Save processed data

Usage:
    python scripts/run_preprocessing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore [missing-import]
from src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
# pyrefly: ignore [missing-import]
from src.utils.seed import set_seed
# pyrefly: ignore [missing-import]
from src.utils.config import load_config


def main() -> None:
    config = load_config()
    set_seed(config["project"]["random_seed"])
    run_preprocessing_pipeline(config)


if __name__ == "__main__":
    main()