"""
validation.py

Purpose: Schema and quality validation of the raw UNSW-NB15 training/testing
data, run BEFORE any cleaning or preprocessing. Writes a human-readable
report to results/reports/data_validation_report.txt.
"""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# pyrefly: ignore [missing-import]
from src.utils.config import load_config, load_classes
# pyrefly: ignore [missing-import]
from src.utils.paths import get_results_dir
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[dict] = None,
    classes: Optional[dict] = None,
) -> Dict:
    """
    Runs all schema/quality checks and returns a dict with:
      - "issues": list of FAIL/WARN strings that need attention
      - "report_lines": the full human-readable report (also logged live)
    """
    config = config or load_config()
    classes = classes or load_classes()
    target_col = config["data"]["target_column"]

    report_lines: List[str] = []
    issues: List[str] = []

    def emit(line: str):
        report_lines.append(line)
        logger.info(line)

    emit("=" * 70)
    emit("NetEvolve — Data Validation Report")
    emit("=" * 70)

    # 1. Target column present in both splits
    for name, df in [("training", train_df), ("testing", test_df)]:
        if target_col not in df.columns:
            msg = f"[FAIL] Target column '{target_col}' missing from {name} set"
            issues.append(msg)
            emit(msg)
        else:
            emit(f"[OK]   Target column '{target_col}' present in {name} set")

    # 2 & 9. Train/test schema compatibility
    train_cols, test_cols = set(train_df.columns), set(test_df.columns)
    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols
    if only_in_train or only_in_test:
        msg = (
            f"[WARN] Column mismatch between train/test — "
            f"only in train: {sorted(only_in_train)}, only in test: {sorted(only_in_test)}"
        )
        issues.append(msg)
        emit(msg)
    else:
        emit("[OK]   Training and testing schemas match")

    # 3. Expected columns present (id + categorical columns from config)
    expected = [config["data"].get("id_column")] + config["preprocessing"]["categorical_columns"]
    expected = [c for c in expected if c]
    for name, df in [("training", train_df), ("testing", test_df)]:
        missing_expected = [c for c in expected if c not in df.columns]
        if missing_expected:
            msg = f"[WARN] Expected columns missing from {name} set: {missing_expected}"
            issues.append(msg)
            emit(msg)
        else:
            emit(f"[OK]   All expected configured columns present in {name} set")

    # 4. Missing values
    for name, df in [("training", train_df), ("testing", test_df)]:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            emit(f"[OK]   No missing values in {name} set")
        else:
            emit(f"[WARN] Missing values in {name} set:\n{missing.to_string()}")

    # 5. Duplicate records
    for name, df in [("training", train_df), ("testing", test_df)]:
        dup_count = int(df.duplicated().sum())
        emit(f"[INFO] Duplicate rows in {name} set: {dup_count}")

    # 6. Data types
    for name, df in [("training", train_df), ("testing", test_df)]:
        emit(f"[INFO] {name} set dtypes:\n{df.dtypes.to_string()}")

    # 7 & 8. Class distribution + known/future-unknown class presence
    if target_col in train_df.columns:
        dist = train_df[target_col].value_counts()
        emit(f"[INFO] Class distribution ('{target_col}') in training set:\n{dist.to_string()}")

        actual_classes = set(train_df[target_col].astype(str).str.strip().unique())
        configured_classes = set(classes["known_classes"]) | set(classes["future_unknown_classes"])

        missing_from_data = configured_classes - actual_classes
        if missing_from_data:
            msg = (
                f"[FAIL] Configured class(es) not found in dataset "
                f"(check exact spelling in config/classes.yaml): {sorted(missing_from_data)}"
            )
            issues.append(msg)
            emit(msg)
        else:
            emit("[OK]   All configured known/future-unknown classes found in the dataset")

        unconfigured = actual_classes - configured_classes
        if unconfigured:
            msg = f"[WARN] Class(es) found in dataset but not listed in classes.yaml: {sorted(unconfigured)}"
            issues.append(msg)
            emit(msg)

    emit("=" * 70)
    emit(f"Validation complete. Issues flagged: {len(issues)}")
    emit("=" * 70)

    return {"issues": issues, "report_lines": report_lines}


def save_report(report_lines: List[str], config: Optional[dict] = None) -> Path:
    config = config or load_config()
    results_dir = get_results_dir(config)
    report_path = results_dir / "reports" / "data_validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Validation report saved to: {report_path}")
    return report_path
