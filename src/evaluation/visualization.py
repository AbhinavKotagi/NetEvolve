"""
visualization.py

Purpose: Confusion matrix + (in a later phase) training-curve plotting,
saved to results/figures/. Kept separate from metrics.py so numeric
computation and plotting can be tested/used independently.
"""
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    confusion_matrix: List[List[int]],
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    cm = np.array(confusion_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to: {save_path}")
    return save_path


def plot_training_history(history: List[dict], save_dir: Path) -> None:
    """Plots Training/Validation Loss vs Epoch and Training/Validation
    Accuracy vs Epoch, saved as two separate figures in save_dir."""
    epochs = [h["epoch"] for h in history]
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="Training Loss")
    ax.plot(epochs, [h["validation_loss"] for h in history], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Epoch")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_dir / "neural_loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_accuracy"] for h in history], label="Training Accuracy")
    ax.plot(epochs, [h["validation_accuracy"] for h in history], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Epoch")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_dir / "neural_accuracy_curve.png", dpi=150)
    plt.close(fig)

    logger.info(f"Training curves saved to: {save_dir}")