"""
baseline.py

Purpose: RandomForestClassifier training/evaluation wrapper. This is the
reference score the neural network (Phase 6) has to beat.
"""
import json
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

# pyrefly: ignore [missing-import]
from src.evaluation.metrics import compute_metrics
# pyrefly: ignore [missing-import]
from src.evaluation.visualization import plot_confusion_matrix
# pyrefly: ignore [missing-import]
from src.models.model_factory import build_baseline_model
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_baseline(X_train, y_train, config: dict) -> RandomForestClassifier:
    model = build_baseline_model(config)
    logger.info(f"Training RandomForestClassifier ({config['baseline_model']['hyperparameters']})")
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(model: RandomForestClassifier, X, y, class_names, split_name: str) -> dict:
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, class_names)
    logger.info(
        f"[{split_name}] Accuracy={metrics['accuracy']:.4f} "
        f"Macro F1={metrics['f1_macro']:.4f} Weighted F1={metrics['f1_weighted']:.4f}"
    )
    logger.info(f"[{split_name}] Classification report:\n{metrics['classification_report']}")
    return metrics


def save_baseline_model(model: RandomForestClassifier, models_dir: Path) -> Path:
    baseline_dir = models_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    path = baseline_dir / "random_forest.joblib"
    joblib.dump(model, path)
    logger.info(f"Baseline model saved to: {path}")
    return path


def save_baseline_metrics(val_metrics: dict, test_metrics: dict, results_dir: Path) -> Path:
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "baseline_metrics.json"

    payload = {
        "validation": {k: v for k, v in val_metrics.items() if k != "classification_report"},
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Baseline metrics saved to: {path}")
    return path


def save_baseline_confusion_matrices(val_metrics: dict, test_metrics: dict, results_dir: Path) -> None:
    figures_dir = results_dir / "figures"
    plot_confusion_matrix(
        val_metrics["confusion_matrix"], val_metrics["class_names"],
        figures_dir / "baseline_confusion_matrix_val.png", title="Baseline (Validation)",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], test_metrics["class_names"],
        figures_dir / "baseline_confusion_matrix_test.png", title="Baseline (Test)",
    )