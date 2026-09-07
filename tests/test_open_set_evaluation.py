"""
test_open_set_evaluation.py

Unit tests for Phase 5 Open-Set Evaluation logic (Youden's index calculation
and threshold-based prediction).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.open_set_evaluator import OpenSetEvaluator


class TestOpenSetEvaluator:
    def test_fit_threshold_perfect_separation(self):
        evaluator = OpenSetEvaluator()
        
        # Perfect separation: all unknowns have higher uncertainty than knowns
        # Knowns (0): u in [0.1, 0.2, 0.3]
        # Unknowns (1): u in [0.7, 0.8, 0.9]
        uncertainties = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        is_unknown = np.array([0, 0, 0, 1, 1, 1])
        
        tau = evaluator.fit_threshold(uncertainties, is_unknown)
        
        # Any threshold strictly between 0.3 and 0.7 gives perfect classification.
        # Scikit-learn's roc_curve thresholds usually pick the exact value of the lowest positive instance (or close to it)
        # So tau should be 0.7.
        assert tau >= 0.3 and tau <= 0.7

    def test_predict_without_fit_raises_error(self):
        evaluator = OpenSetEvaluator()
        beliefs = np.array([[0.8, 0.2], [0.3, 0.7]])
        uncertainties = np.array([0.1, 0.9])
        
        with pytest.raises(ValueError, match="Threshold not provided and model not fitted"):
            evaluator.predict(beliefs, uncertainties)

    def test_predict_with_threshold(self):
        evaluator = OpenSetEvaluator()
        
        # K = 3 known classes (0, 1, 2)
        # B = 4 samples
        beliefs = np.array([
            [0.8, 0.1, 0.1],  # Pred class 0
            [0.1, 0.7, 0.2],  # Pred class 1
            [0.2, 0.2, 0.6],  # Pred class 2
            [0.3, 0.3, 0.4],  # Pred class 2 (but highly uncertain)
        ])
        
        uncertainties = np.array([0.1, 0.2, 0.3, 0.8])
        
        # Set threshold to 0.7 -> Sample 3 should be classified as Unknown (K=3)
        preds = evaluator.predict(beliefs, uncertainties, threshold=0.7)
        
        expected_preds = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(preds, expected_preds)

    def test_predict_uses_fitted_threshold(self):
        evaluator = OpenSetEvaluator()
        evaluator.optimal_threshold = 0.5
        
        beliefs = np.array([
            [0.9, 0.1],  # K=2 classes
            [0.4, 0.6],
        ])
        uncertainties = np.array([0.2, 0.6]) # 2nd one is >= 0.5, so unknown
        
        preds = evaluator.predict(beliefs, uncertainties)
        
        expected_preds = np.array([0, 2])
        np.testing.assert_array_equal(preds, expected_preds)
