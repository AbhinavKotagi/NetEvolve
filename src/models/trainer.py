"""
trainer.py

Purpose: Training loop for NeuralClassifier — AdamW, CrossEntropyLoss, early
stopping on validation Macro F1, best-checkpoint saving, and training-history
tracking.
"""
import json
from pathlib import Path
from typing import Dict, List

import torch

import torch.nn as nn

from torch.optim import AdamW

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score



from utils.logger import get_logger

logger = get_logger(__name__)


def get_device(config: dict) -> torch.device:
    use_cuda = config.get("device", {}).get("use_cuda_if_available", True)
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. If optimizer is given, trains; otherwise evaluates."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1


def train_neural_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    models_dir: Path,
) -> Dict:
    """
    Trains `model`, tracking train/val loss+accuracy and val macro F1 each
    epoch. Applies early stopping on validation macro F1 and saves the best
    checkpoint to models/neural/best_model.pt (not the final epoch).

    Returns the training history dict (also written by the caller if desired).
    """
    device = get_device(config)
    model.to(device)
    logger.info(f"Training on device: {device}")

    train_cfg = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    patience = train_cfg["early_stopping_patience"]
    epochs = train_cfg["epochs"]

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history: List[Dict] = []

    neural_dir = models_dir / "neural"
    neural_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = neural_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_f1 = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | "
            f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f} | "
            f"Validation Macro F1: {val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": model.input_dim,
                "embedding_dimension": model.embedding_dimension,
                "num_classes": model.num_classes,
                "epoch": epoch,
                "validation_macro_f1": val_f1,
            }, checkpoint_path)
            logger.info(f"[CHECKPOINT] New best model (Val Macro F1={val_f1:.4f}) saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    f"[EARLY STOP] No improvement for {patience} epochs. "
                    f"Stopping at epoch {epoch} (best Val Macro F1={best_val_f1:.4f})."
                )
                break

    return {"history": history, "best_validation_f1": best_val_f1, "checkpoint_path": str(checkpoint_path)}


def save_training_history(history: List[Dict], results_dir: Path) -> Path:
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "training_history.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {path}")
    return path


def load_best_checkpoint(checkpoint_path: Path, model: nn.Module) -> nn.Module:
    """Loads the saved best-checkpoint weights into an already-constructed
    model instance (the caller must build the model with matching dims first)."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Neural network checkpoint not found: {checkpoint_path}. "
            "Run `python scripts/train_neural_model.py` first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model