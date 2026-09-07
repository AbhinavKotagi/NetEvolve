"""
test_opinion_generator.py

Unit tests for Phase 3 opinion generation:
- SingleViewOpinionGenerator: softplus usage, Dirichlet invariants, alpha = e+1
- MultiViewOpinionGenerator: applies to all three views
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.opinion_generator import SingleViewOpinionGenerator, MultiViewOpinionGenerator


class TestSingleViewOpinionGenerator:
    def test_output_shapes_and_invariants(self):
        B = 4
        feature_dim = 64
        num_classes = 5

        generator = SingleViewOpinionGenerator(feature_dim, num_classes)
        x = torch.randn(B, feature_dim)

        out = generator(x)

        assert "evidence" in out
        assert "alpha" in out
        assert "belief" in out
        assert "uncertainty" in out
        assert "S" in out

        e = out["evidence"]
        alpha = out["alpha"]
        b = out["belief"]
        u = out["uncertainty"]
        S = out["S"]

        # Shapes
        assert e.shape == (B, num_classes)
        assert alpha.shape == (B, num_classes)
        assert b.shape == (B, num_classes)
        assert u.shape == (B, 1)
        assert S.shape == (B, 1)

        # Invariants
        # Evidence must be positive (softplus)
        assert torch.all(e > 0)
        # Alpha = e + 1
        assert torch.allclose(alpha, e + 1.0)
        # S = sum(alpha)
        assert torch.allclose(S, torch.sum(alpha, dim=1, keepdim=True))
        # Belief = e / S
        assert torch.allclose(b, e / S)
        # Uncertainty = K / S
        assert torch.allclose(u, num_classes / S)
        # Subjective logic constraint: sum(b) + u = 1
        assert torch.allclose(torch.sum(b, dim=1, keepdim=True) + u, torch.ones(B, 1))

    def test_gradients_flow(self):
        generator = SingleViewOpinionGenerator(64, 5)
        x = torch.randn(2, 64, requires_grad=True)
        out = generator(x)
        loss = out["evidence"].sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_softplus_activation(self):
        # We can test that the evidence output is exactly softplus(W*x+b)
        generator = SingleViewOpinionGenerator(64, 5)
        x = torch.randn(2, 64)

        out = generator(x)
        expected_e = F.softplus(generator.fc(x))

        assert torch.allclose(out["evidence"], expected_e)


class TestMultiViewOpinionGenerator:
    def test_output_structure(self):
        generator = MultiViewOpinionGenerator(64, 5)

        features = {
            "ip_features": torch.randn(2, 64),
            "transport_features": torch.randn(2, 64),
            "payload_features": torch.randn(2, 64),
        }

        out = generator(features)

        assert "ip" in out
        assert "transport" in out
        assert "payload" in out

        for view in ["ip", "transport", "payload"]:
            assert "evidence" in out[view]
            assert "alpha" in out[view]
            assert out[view]["evidence"].shape == (2, 5)

    def test_independent_weights(self):
        generator = MultiViewOpinionGenerator(64, 5)
        ip_params = {id(p) for p in generator.ip_opinion.parameters()}
        tr_params = {id(p) for p in generator.transport_opinion.parameters()}
        pay_params = {id(p) for p in generator.payload_opinion.parameters()}

        assert len(ip_params & tr_params) == 0
        assert len(ip_params & pay_params) == 0
        assert len(tr_params & pay_params) == 0
