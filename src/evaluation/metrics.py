"""
metrics.py

Purpose: Shared metric computation used by BOTH the baseline and (in a later
phase) the neural network, so the two models are always scored identically.
"""
from typing import List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, class_names: Optional[List[str]] = None) -> dict:
    """Returns accuracy, macro/weighted precision-recall-F1, per-class
    precision/recall/F1, a text classification report, and the confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    labels = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    names = class_names if class_names is not None else [str(l) for l in labels]

    per_class = {
        names[i]: {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support_per_class[i]),
        }
        for i in range(len(names))
    }

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, target_names=names, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names": names,
    }