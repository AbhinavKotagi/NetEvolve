"""
open_set_evaluator.py

Purpose: Implements Open-Set evaluation using uncertainty thresholding.
Implements §3.5 / Section III-E ("Novel Class Discovery") of Wang et al. 2025.

Calculates the optimal uncertainty threshold (tau) using Youden's Index on a
known/unknown validation set, then applies it to classify test samples as
Known or Unknown.

For Known samples, it falls back to the max belief mass (argmax b_k) for closed-set
classification.

RoNeTC Phase 5.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve
from typing import Dict, Tuple, List
import torch

class OpenSetEvaluator:
    """Evaluates RoNeTC outputs for Open-Set Recognition (OSR)."""

    def __init__(self) -> None:
        self.optimal_threshold: float = -1.0

    def fit_threshold(
        self,
        uncertainties: np.ndarray,
        is_unknown: np.ndarray,
    ) -> float:
        """
        Calculates the optimal uncertainty threshold (tau) using Youden's Index.

        Parameters
        ----------
        uncertainties : np.ndarray
            1D array of uncertainty scores (u) from the fused opinion.
        is_unknown : np.ndarray
            1D boolean or int array where 1 (or True) indicates an Unknown class sample,
            and 0 (or False) indicates a Known class sample.

        Returns
        -------
        float
            The optimal threshold value (tau). Samples with u >= tau are classified as Unknown.
        """
        # ROC curve computes thresholds for classifying as '1' (Unknown)
        fpr, tpr, thresholds = roc_curve(is_unknown, uncertainties)

        # Youden's J statistic = TPR + TNR - 1 = TPR - FPR
        youden_j = tpr - fpr

        # Find the threshold that maximizes Youden's J
        optimal_idx = np.argmax(youden_j)
        self.optimal_threshold = float(thresholds[optimal_idx])
        
        return self.optimal_threshold

    def predict(
        self,
        beliefs: np.ndarray,
        uncertainties: np.ndarray,
        threshold: float = None,
    ) -> np.ndarray:
        """
        Predicts class labels including the 'Unknown' class.

        Parameters
        ----------
        beliefs : np.ndarray
            2D array of shape (N, K) containing belief masses for K known classes.
        uncertainties : np.ndarray
            1D array of uncertainty scores (u).
        threshold : float, optional
            The uncertainty threshold. If None, uses the fitted optimal_threshold.

        Returns
        -------
        np.ndarray
            1D array of integer predictions.
            0 to K-1 represent the known classes.
            K represents the 'Unknown' class.
        """
        if threshold is None:
            if self.optimal_threshold < 0:
                raise ValueError("Threshold not provided and model not fitted.")
            threshold = self.optimal_threshold

        K = beliefs.shape[1]
        
        # Closed-set predictions (argmax of beliefs)
        closed_set_preds = np.argmax(beliefs, axis=1)

        # Apply threshold to find unknowns
        is_unknown_pred = uncertainties >= threshold

        # Final predictions: K for unknown, argmax for known
        final_preds = np.where(is_unknown_pred, K, closed_set_preds)

        return final_preds
