"""
train_baseline.py

CLI entry point:
  Load processed data -> Train Random Forest -> Evaluate -> Save model

Usage:
    python scripts/train_baseline.py
"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore [missing-import]
from src.models.baseline import (
    train_baseline,
    evaluate_baseline,
    save_baseline_model,
    save_baseline_metrics,
    save_baseline_confusion_matrices,
)
# pyrefly: ignore [missing-import]
from src.utils.config import load_config
# pyrefly: ignore [missing-import]
from src.utils.paths import get_processed_data_dir, get_models_dir, get_results_dir
# pyrefly: ignore [missing-import]
from src.utils.seed import set_seed
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    set_seed(config["project"]["random_seed"])

    processed_dir = get_processed_data_dir(config)
    models_dir = get_models_dir(config)
    results_dir = get_results_dir(config)

    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                "X_test_known.npy", "y_test_known.npy", "processed_metadata.json"]
    missing = [f for f in required if not (processed_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed file(s) {missing} in {processed_dir}. "
            "Run `python scripts/run_preprocessing.py` first."
        )

    X_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    X_test = np.load(processed_dir / "X_test_known.npy")
    y_test = np.load(processed_dir / "y_test_known.npy")

    with open(processed_dir / "processed_metadata.json") as f:
        metadata = json.load(f)
    class_names = [cls for cls, _ in sorted(metadata["class_mapping"].items(), key=lambda kv: kv[1])]

    model = train_baseline(X_train, y_train, config)

    val_metrics = evaluate_baseline(model, X_val, y_val, class_names, "Validation")
    test_metrics = evaluate_baseline(model, X_test, y_test, class_names, "Test (known)")

    save_baseline_model(model, models_dir)
    save_baseline_metrics(val_metrics, test_metrics, results_dir)
    save_baseline_confusion_matrices(val_metrics, test_metrics, results_dir)


if __name__ == "__main__":
    main()