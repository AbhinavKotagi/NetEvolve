"""
visualization.py

Purpose: Confusion matrix + (in a later phase) training-curve plotting,
saved to results/figures/. Kept separate from metrics.py so numeric
computation and plotting can be tested/used independently.
"""
from pathlib import Path
from typing import List
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

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