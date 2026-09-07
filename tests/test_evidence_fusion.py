"""
test_evidence_fusion.py

Unit tests for Dempster-Shafer Evidence Combination Rule.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.evidence_fusion import DempsterShaferFusion


class TestDempsterShaferFusion:
    def test_single_opinion_returns_itself(self):
        fusion = DempsterShaferFusion()
        op = {
            "belief": torch.tensor([[0.5, 0.3]]),
            "uncertainty": torch.tensor([[0.2]]),
        }
        res = fusion([op])
        assert torch.allclose(res["belief"], op["belief"])
        assert torch.allclose(res["uncertainty"], op["uncertainty"])

    def test_two_opinions_fusion(self):
        fusion = DempsterShaferFusion()
        op1 = {
            "belief": torch.tensor([[0.4, 0.1]]),
            "uncertainty": torch.tensor([[0.5]]),
        }
        op2 = {
            "belief": torch.tensor([[0.2, 0.3]]),
            "uncertainty": torch.tensor([[0.5]]),
        }

        # Calculate expected values manually
        # b1 = [0.4, 0.1], u1 = 0.5
        # b2 = [0.2, 0.3], u2 = 0.5
        # C = b1[0]*b2[1] + b1[1]*b2[0] = 0.4*0.3 + 0.1*0.2 = 0.12 + 0.02 = 0.14
        # denom = 1 - 0.14 = 0.86
        # bf[0] = (b1[0]*b2[0] + b1[0]*u2 + b2[0]*u1) / denom
        #       = (0.08 + 0.2 + 0.1) / 0.86 = 0.38 / 0.86 = 0.44186
        # bf[1] = (b1[1]*b2[1] + b1[1]*u2 + b2[1]*u1) / denom
        #       = (0.03 + 0.05 + 0.15) / 0.86 = 0.23 / 0.86 = 0.26744
        # uf = (u1 * u2) / denom = 0.25 / 0.86 = 0.29070

        expected_b = torch.tensor([[0.38 / 0.86, 0.23 / 0.86]])
        expected_u = torch.tensor([[0.25 / 0.86]])

        res = fusion([op1, op2])

        assert torch.allclose(res["belief"], expected_b, atol=1e-5)
        assert torch.allclose(res["uncertainty"], expected_u, atol=1e-5)

        # Check invariants
        assert torch.allclose(res["belief"].sum(dim=1, keepdim=True) + res["uncertainty"], torch.ones(1, 1))

    def test_complete_conflict_is_clamped(self):
        fusion = DempsterShaferFusion(conflict_clamp_min=1e-5)
        # Opinion 1 is 100% sure it's class 0
        op1 = {
            "belief": torch.tensor([[1.0, 0.0]]),
            "uncertainty": torch.tensor([[0.0]]),
        }
        # Opinion 2 is 100% sure it's class 1
        op2 = {
            "belief": torch.tensor([[0.0, 1.0]]),
            "uncertainty": torch.tensor([[0.0]]),
        }

        # C will be 1.0. Denominator will be clamped to 1e-5.
        res = fusion([op1, op2])

        # Since b1[k]*b2[k] = 0, b1[k]*u2 = 0, b2[k]*u1 = 0, fused belief is 0
        assert torch.allclose(res["belief"], torch.zeros(1, 2))
        assert torch.allclose(res["uncertainty"], torch.zeros(1, 1))
        # Mathematically, complete conflict is undefined without clamping, but clamping ensures no NaN.

    def test_associativity_three_opinions(self):
        fusion = DempsterShaferFusion()
        op1 = {"belief": torch.tensor([[0.3, 0.2]]), "uncertainty": torch.tensor([[0.5]])}
        op2 = {"belief": torch.tensor([[0.1, 0.4]]), "uncertainty": torch.tensor([[0.5]])}
        op3 = {"belief": torch.tensor([[0.6, 0.1]]), "uncertainty": torch.tensor([[0.3]])}

        res_12_3 = fusion([fusion([op1, op2]), op3])
        res_1_23 = fusion([op1, fusion([op2, op3])])

        assert torch.allclose(res_12_3["belief"], res_1_23["belief"], atol=1e-5)
        assert torch.allclose(res_12_3["uncertainty"], res_1_23["uncertainty"], atol=1e-5)

    def test_alpha_and_evidence_reconstruction(self):
        fusion = DempsterShaferFusion()
        op1 = {"belief": torch.tensor([[0.4, 0.1]]), "uncertainty": torch.tensor([[0.5]])}
        op2 = {"belief": torch.tensor([[0.2, 0.3]]), "uncertainty": torch.tensor([[0.5]])}

        res = fusion([op1, op2])

        K = 2
        u = res["uncertainty"]
        b = res["belief"]

        S = K / u
        expected_e = b * S
        expected_alpha = expected_e + 1.0

        assert torch.allclose(res["S"], S)
        assert torch.allclose(res["evidence"], expected_e)
        assert torch.allclose(res["alpha"], expected_alpha)
