"""
test_ronetc_loss.py

Unit tests for the RoNeTC Evidential Deep Learning loss function.
Checks Bayes-risk CE computation, KL divergence term, and annealing logic.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from losses.ronetc_loss import RoNeTCLoss, calculate_annealing_factor


class TestRoNeTCLoss:
    def test_loss_shape_and_positivity(self):
        B = 4
        K = 5
        loss_fn = RoNeTCLoss(num_classes=K)

        # Alpha >= 1
        alpha = torch.rand(B, K) + 1.0
        y = torch.randint(0, K, (B,))

        loss = loss_fn(alpha, y, lambda_t=0.5)

        assert loss.shape == (B,)
        assert torch.all(loss > 0)

    def test_kl_divergence_is_zero_for_uniform_alpha(self):
        B = 2
        K = 5
        loss_fn = RoNeTCLoss(num_classes=K)

        # If alpha is uniform (all ones), KL to Dirichlet(1) should be 0
        alpha_uniform = torch.ones(B, K)

        kl = loss_fn.kl_divergence(alpha_uniform)

        assert torch.allclose(kl, torch.zeros_like(kl))

    def test_kl_divergence_is_positive_for_non_uniform_alpha(self):
        B = 2
        K = 5
        loss_fn = RoNeTCLoss(num_classes=K)

        alpha = torch.ones(B, K)
        # Make it non-uniform
        alpha[0, 0] = 5.0
        alpha[1, 2] = 10.0

        kl = loss_fn.kl_divergence(alpha)

        assert torch.all(kl > 0)

    def test_gradients_flow(self):
        B = 2
        K = 3
        loss_fn = RoNeTCLoss(num_classes=K)

        alpha = (torch.rand(B, K) + 1.0).requires_grad_()
        y = torch.randint(0, K, (B,))

        loss = loss_fn(alpha, y, lambda_t=1.0)
        total_loss = loss.mean()
        total_loss.backward()

        assert alpha.grad is not None
        assert alpha.grad.abs().sum() > 0

    def test_loss_decreases_with_better_predictions(self):
        # A prediction that matches the ground truth should have lower loss
        # than a prediction that contradicts it.
        B = 1
        K = 3
        loss_fn = RoNeTCLoss(num_classes=K)

        y = torch.tensor([0])

        # Good prediction (high evidence for class 0, low for others)
        alpha_good = torch.tensor([[10.0, 1.1, 1.1]])
        loss_good = loss_fn(alpha_good, y, lambda_t=0.5).item()

        # Bad prediction (high evidence for class 1, low for others)
        alpha_bad = torch.tensor([[1.1, 10.0, 1.1]])
        loss_bad = loss_fn(alpha_bad, y, lambda_t=0.5).item()

        assert loss_good < loss_bad


def test_calculate_annealing_factor():
    # Linear ramp up to annealing_epochs, then caps at annealing_ceiling
    assert calculate_annealing_factor(0, 10) == 0.0
    assert calculate_annealing_factor(5, 10) == 0.5
    assert calculate_annealing_factor(10, 10) == 1.0
    assert calculate_annealing_factor(15, 10) == 1.0

    # With a different ceiling
    assert calculate_annealing_factor(5, 10, 0.8) == 0.4
    assert calculate_annealing_factor(10, 10, 0.8) == 0.8
    assert calculate_annealing_factor(15, 10, 0.8) == 0.8
