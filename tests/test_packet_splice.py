"""
test_packet_splice.py

Unit tests for packet_splice.py:
    - compute_grid_shape for various r values
    - PacketSplice output shape for exact-divisible and non-divisible l
    - Determinism and gradient flow
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.packet_splice import PacketSplice, compute_grid_shape


# ===========================================================================
# compute_grid_shape
# ===========================================================================

class TestComputeGridShape:
    def test_r4(self):
        assert compute_grid_shape(4) == (2, 2)

    def test_r1(self):
        assert compute_grid_shape(1) == (1, 1)

    def test_r3_non_square(self):
        g_h, g_w = compute_grid_shape(3)
        assert g_h * g_w == 3
        assert g_h <= g_w

    def test_r6(self):
        g_h, g_w = compute_grid_shape(6)
        assert g_h * g_w == 6
        assert g_h <= g_w

    def test_r8(self):
        g_h, g_w = compute_grid_shape(8)
        assert g_h * g_w == 8

    def test_product_equals_r(self):
        for r in [1, 2, 3, 4, 6, 8, 9, 12, 16]:
            g_h, g_w = compute_grid_shape(r)
            assert g_h * g_w == r, f"r={r}: g_h*g_w={g_h*g_w} != {r}"

    def test_g_h_le_g_w(self):
        for r in [1, 2, 3, 4, 6, 8, 9, 12, 16]:
            g_h, g_w = compute_grid_shape(r)
            assert g_h <= g_w, f"r={r}: g_h={g_h} > g_w={g_w}"

    def test_deterministic(self):
        for r in [4, 8, 12]:
            assert compute_grid_shape(r) == compute_grid_shape(r)


# ===========================================================================
# PacketSplice output shape
# ===========================================================================

class TestPacketSplice:
    def _x(self, B=2, l=12, H_p=8, W_p=8):
        return torch.rand(B, l, H_p, W_p)

    def test_l12_r4_output_shape(self):
        splice = PacketSplice(r=4)
        x = self._x(l=12)
        out = splice(x)
        # C = 12/4 = 3, g_h=g_w=2 → H=2*8=16, W=2*8=16
        g_h, g_w = compute_grid_shape(4)
        assert out.shape == (2, 3, g_h * 8, g_w * 8)

    def test_l8_r4_output_shape(self):
        splice = PacketSplice(r=4)
        x = self._x(l=8)
        out = splice(x)
        g_h, g_w = compute_grid_shape(4)
        assert out.shape == (2, 2, g_h * 8, g_w * 8)

    def test_non_divisible_padded_to_next_multiple(self):
        """l=10, r=4 → padded to 12, C=3."""
        splice = PacketSplice(r=4)
        x = self._x(l=10)
        out = splice(x)
        g_h, g_w = compute_grid_shape(4)
        assert out.shape == (2, 3, g_h * 8, g_w * 8)

    def test_l4_r4_single_channel(self):
        splice = PacketSplice(r=4)
        x = self._x(l=4)
        out = splice(x)
        g_h, g_w = compute_grid_shape(4)
        assert out.shape == (2, 1, g_h * 8, g_w * 8)

    def test_r1_each_packet_is_own_channel(self):
        splice = PacketSplice(r=1)
        x = self._x(l=5)
        out = splice(x)
        # r=1 → g_h=1, g_w=1, H=H_p, W=W_p, C=5
        assert out.shape == (2, 5, 8, 8)

    def test_output_spatial_size_helper(self):
        splice = PacketSplice(r=4)
        C, H, W = splice.output_spatial_size(H_p=8, W_p=8, l=12)
        assert C == 3
        g_h, g_w = compute_grid_shape(4)
        assert H == g_h * 8
        assert W == g_w * 8

    def test_values_in_range_preserved(self):
        """Splice should not change values outside [0,1] if input is in [0,1]."""
        splice = PacketSplice(r=4)
        x = torch.rand(2, 4, 8, 8)
        out = splice(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_padding_value_appears_in_padded_channel(self):
        """Extra padded packets should have padding_value."""
        splice = PacketSplice(r=4, padding_value=0.0)
        # l=4, r=4 → C=1, no extra padding; use l=2, r=4 → padded to l=4, C=1
        x = torch.ones(1, 2, 4, 4)
        out = splice(x)
        # Padded packets are 0.0; the original 2 packets are 1.0
        # After grid arrangement: half the pixels should be 1.0, half 0.0
        assert out.min() == 0.0
        assert out.max() == 1.0

    def test_batch_size_independence(self):
        splice = PacketSplice(r=4)
        x1 = torch.rand(1, 12, 8, 8)
        x8 = torch.rand(8, 12, 8, 8)
        o1 = splice(x1)
        o8 = splice(x8)
        assert o1.shape[1:] == o8.shape[1:]

    def test_gradient_flows(self):
        splice = PacketSplice(r=4)
        x = torch.rand(2, 12, 8, 8, requires_grad=True)
        out = splice(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_deterministic(self):
        splice = PacketSplice(r=4)
        x = torch.rand(2, 12, 8, 8)
        o1 = splice(x)
        o2 = splice(x)
        assert torch.allclose(o1, o2)
