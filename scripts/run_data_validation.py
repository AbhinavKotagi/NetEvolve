"""
run_data_validation.py

CLI entry point:
  Load data -> Validate schema -> Generate validation report

Usage:
    python scripts/run_data_validation.py
"""
import sys
from pathlib import Path

# Allow running as `python scripts/run_data_validation.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.loader import load_training_data, load_testing_data
from data.validation import validate_dataset, save_report
from utils.config import load_config, load_classes
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    classes = load_classes()

    train_df = load_training_data(config)
    test_df = load_testing_data(config)

    result = validate_dataset(train_df, test_df, config, classes)
    save_report(result["report_lines"], config)

    if result["issues"]:
        logger.warning(
            f"Validation finished with {len(result['issues'])} issue(s) — "
            "review results/reports/data_validation_report.txt before proceeding."
        )
        sys.exit(1)
    else:
        logger.info("Validation finished with no issues.")


if __name__ == "__main__":
    main()
