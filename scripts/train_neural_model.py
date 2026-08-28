"""
train_neural_model.py

CLI entry point:
  Load processed data -> Create DataLoaders -> Create neural model -> Train
  -> Early stopping -> Save best checkpoint -> Evaluate -> Save results
  -> Compare against the Random Forest baseline

Usage:
    python scripts/train_neural_model.py
"""
import sys
from pathlib import Path
import json
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import build_dataloaders
from src.models.model_factory import build_neural_model
from src.models.trainer import train_neural_model, save_training_history, get_device
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualization import plot_confusion_matrix, plot_training_history
from src.utils.config import load_config
from src.utils.paths import get_processed_data_dir, get_models_dir, get_results_dir
from src.utils.seed import set_seed
from src.utils.logger import get_logger

import torch

logger = get_logger(__name__)


def _load_processed(processed_dir: Path):
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

    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y_batch.tolist())
    return all_labels, all_preds


def main() -> None:
    config = load_config()
    set_seed(config["project"]["random_seed"])

    processed_dir = get_processed_data_dir(config)
    models_dir = get_models_dir(config)
    results_dir = get_results_dir(config)

    X_train, y_train, X_val, y_val, X_test, y_test, metadata = _load_processed(processed_dir)
    class_names = [cls for cls, _ in sorted(metadata["class_mapping"].items(), key=lambda kv: kv[1])]
    num_classes = len(class_names)
    input_dim = metadata["n_features"]

    if X_train.shape[1] != input_dim:
        raise ValueError(
            f"Feature dimension mismatch: processed_metadata.json says {input_dim}, "
            f"but X_train.npy has {X_train.shape[1]} columns. Re-run preprocessing."
        )

    train_loader, val_loader, test_loader = build_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config["training"]["batch_size"],
    )

    model = build_neural_model(config, input_dim=input_dim, num_classes=num_classes)

    result = train_neural_model(model, train_loader, val_loader, config, models_dir)
    save_training_history(result["history"], results_dir)
    plot_training_history(result["history"], results_dir / "figures")

    # Reload the BEST checkpoint (not necessarily the last epoch's in-memory weights)
    device = get_device(config)
    checkpoint = torch.load(result["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    y_val_true, y_val_pred = _predict(model, val_loader, device)
    y_test_true, y_test_pred = _predict(model, test_loader, device)

    val_metrics = compute_metrics(y_val_true, y_val_pred, class_names)
    test_metrics = compute_metrics(y_test_true, y_test_pred, class_names)

    logger.info(
        f"[Neural — Validation] Accuracy={val_metrics['accuracy']:.4f} "
        f"Macro F1={val_metrics['f1_macro']:.4f} Weighted F1={val_metrics['f1_weighted']:.4f}"
    )
    logger.info(f"[Neural — Validation] Classification report:\n{val_metrics['classification_report']}")
    logger.info(
        f"[Neural — Test (known)] Accuracy={test_metrics['accuracy']:.4f} "
        f"Macro F1={test_metrics['f1_macro']:.4f} Weighted F1={test_metrics['f1_weighted']:.4f}"
    )
    logger.info(f"[Neural — Test (known)] Classification report:\n{test_metrics['classification_report']}")

    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "neural_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "validation": {k: v for k, v in val_metrics.items() if k != "classification_report"},
            "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        }, f, indent=2)

    figures_dir = results_dir / "figures"
    plot_confusion_matrix(
        val_metrics["confusion_matrix"], val_metrics["class_names"],
        figures_dir / "neural_confusion_matrix_val.png", title="Neural Network (Validation)",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], test_metrics["class_names"],
        figures_dir / "neural_confusion_matrix_test.png", title="Neural Network (Test)",
    )

    neural_dir = models_dir / "neural"
    model_metadata = {
        "project_name": config["project"]["name"],
        "dataset": metadata["dataset_source"],
        "known_classes": class_names,
        "input_feature_dimension": input_dim,
        "embedding_dimension": config["neural_model"]["embedding_dimension"],
        "number_of_classes": num_classes,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "best_validation_f1": result["best_validation_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["f1_macro"],
    }
    with open(neural_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)
    logger.info(f"Model metadata saved to: {neural_dir / 'model_metadata.json'}")

    baseline_metrics_path = metrics_dir / "baseline_metrics.json"
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)
        comparison = {
            "Random Forest Baseline": {
                "accuracy": baseline_metrics["test"]["accuracy"],
                "macro_f1": baseline_metrics["test"]["f1_macro"],
                "weighted_f1": baseline_metrics["test"]["f1_weighted"],
            },
            "Neural Network": {
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["f1_macro"],
                "weighted_f1": test_metrics["f1_weighted"],
            },
        }
        with open(metrics_dir / "model_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Model comparison (test set): {json.dumps(comparison, indent=2)}")
    else:
        logger.warning(
            "results/metrics/baseline_metrics.json not found — skipping model comparison. "
            "Run `python scripts/train_baseline.py` to enable it."
        )


if __name__ == "__main__":
    main()
