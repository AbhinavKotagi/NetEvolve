"""
open_set_visualization.py

Purpose: Plotting and visualization tools for open-set evaluation.
Includes density plots for uncertainty scores comparing known vs unknown classes.

RoNeTC Phase 5.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_uncertainty_distribution(
    known_uncertainties: np.ndarray,
    unknown_uncertainties: np.ndarray,
    threshold: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plots the density distribution of uncertainty scores for Known vs Unknown samples.

    Parameters
    ----------
    known_uncertainties : np.ndarray
        Uncertainty scores for known class samples.
    unknown_uncertainties : np.ndarray
        Uncertainty scores for unknown class samples.
    threshold : float, optional
        The calculated decision threshold (tau). If provided, plots a vertical line.
    save_path : Path, optional
        If provided, saves the plot to this path. Otherwise, displays it.
    """
    plt.figure(figsize=(10, 6))

    sns.kdeplot(
        known_uncertainties,
        fill=True,
        label="Known Classes",
        color="blue",
        alpha=0.5,
    )
    sns.kdeplot(
        unknown_uncertainties,
        fill=True,
        label="Unknown Classes",
        color="red",
        alpha=0.5,
    )

    if threshold is not None:
        plt.axvline(
            x=threshold,
            color="black",
            linestyle="--",
            label=f"Threshold (tau={threshold:.3f})",
        )

    plt.title("Uncertainty Score Distribution (Known vs Unknown)", fontsize=14)
    plt.xlabel("Uncertainty Score (u)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
