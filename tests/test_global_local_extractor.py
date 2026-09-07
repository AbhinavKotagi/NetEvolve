"""
test_global_local_extractor.py

Unit tests for Phase 2 components:
    - LocalFeatureRepresentation output shape
    - GlobalFeatureRepresentation unfold→fold round-trip shape consistency
    - FeatureFusion output shape
    - GlobalLocalFeatureExtractor end-to-end shape, determinism, gradients
    - MultiViewFeatureExtractor dict output shape
    - Config-driven parameter changes propagate
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.global_local_extractor import (
    LocalFeatureRepresentation,
    GlobalFeatureRepresentation,
    FeatureFusion,
    GlobalLocalFeatureExtractor,
    MultiViewFeatureExtractor,
)
from models.packet_splice import compute_grid_shape


# ===========================================================================
# Helpers
# ===========================================================================

def _transformer_cfg(num_layers=2, num_heads=4, ff_dim=64, dropout=0.0):
    return {"num_layers": num_layers, "num_heads": num_heads,
            "ff_dim": ff_dim, "dropout": dropout}

def _make_config(
    l=12, r=4, D=32, p=2, feature_dim=64, pooling="avg",
    max_bytes=64,
):
    return {
        "ronetc": {
            "flow": {"packets_per_flow": l},
            "views": {
                "ip_header":        {"max_bytes": max_bytes},
                "transport_header": {"max_bytes": max_bytes},
                "payload":          {"max_bytes": max_bytes},
            },
            "global_local_extractor": {
                "enabled": True,
                "packets_per_channel": r,
                "patch_size": p,
                "local_conv_channels": D,
                "transformer": _transformer_cfg(),
                "pooling": pooling,
                "feature_dim": feature_dim,
            },
        }
    }


# ===========================================================================
# LocalFeatureRepresentation
# ===========================================================================

class TestLocalFeatureRepresentation:
    def test_output_shape(self):
        C, D = 3, 32
        model = LocalFeatureRepresentation(in_channels=C, out_channels=D)
        x = torch.rand(2, C, 16, 16)
        out = model(x)
        assert out.shape == (2, D, 16, 16)

    def test_spatial_dims_preserved(self):
        model = LocalFeatureRepresentation(in_channels=4, out_channels=16)
        for H, W in [(8, 8), (16, 32), (4, 4)]:
            x = torch.rand(1, 4, H, W)
            out = model(x)
            assert out.shape == (1, 16, H, W)

    def test_gradient_flows(self):
        model = LocalFeatureRepresentation(in_channels=3, out_channels=16)
        x = torch.rand(2, 3, 8, 8, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


# ===========================================================================
# GlobalFeatureRepresentation — unfold/fold round-trip
# ===========================================================================

class TestGlobalFeatureRepresentation:
    def _make(self, D=32, p=2, H=8, W=8, layers=1, heads=4, ff=64):
        return GlobalFeatureRepresentation(
            channels=D, patch_size=p,
            num_layers=layers, num_heads=heads, ff_dim=ff, dropout=0.0,
            H=H, W=W,
        )

    def test_output_shape_same_as_input(self):
        model = self._make(D=32, p=2, H=8, W=8)
        x = torch.rand(2, 32, 8, 8)
        out = model(x)
        assert out.shape == x.shape

    def test_various_spatial_sizes(self):
        for H, W in [(4, 4), (8, 8), (6, 6)]:
            model = self._make(D=32, p=2, H=H, W=W)
            x = torch.rand(1, 32, H, W)
            out = model(x)
            assert out.shape == x.shape

    def test_raises_on_non_divisible_spatial(self):
        with pytest.raises(ValueError):
            GlobalFeatureRepresentation(
                channels=32, patch_size=3,
                num_layers=1, num_heads=4, ff_dim=64, dropout=0.0,
                H=8, W=8,   # 8 % 3 != 0
            )

    def test_gradient_flows(self):
        model = self._make()
        x = torch.rand(2, 32, 8, 8, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_size_independence(self):
        model = self._make()
        x1 = torch.rand(1, 32, 8, 8)
        x8 = torch.rand(8, 32, 8, 8)
        o1 = model(x1)
        o8 = model(x8)
        assert o1.shape[1:] == o8.shape[1:]


# ===========================================================================
# FeatureFusion
# ===========================================================================

class TestFeatureFusion:
    def test_output_shape(self):
        D, C = 32, 3
        fuse = FeatureFusion(global_channels=D, local_channels=C)
        gf = torch.rand(2, D, 8, 8)
        skip = torch.rand(2, C, 8, 8)
        out = fuse(gf, skip)
        assert out.shape == (2, C, 8, 8)

    def test_gradient_flows(self):
        fuse = FeatureFusion(global_channels=32, local_channels=3)
        gf = torch.rand(2, 32, 8, 8, requires_grad=True)
        skip = torch.rand(2, 3, 8, 8, requires_grad=True)
        fuse(gf, skip).sum().backward()
        assert gf.grad is not None
        assert skip.grad is not None


# ===========================================================================
# GlobalLocalFeatureExtractor — end-to-end
# ===========================================================================

class TestGlobalLocalFeatureExtractor:
    def _make(self, l=12, H_p=8, W_p=8, r=4, D=32, p=2, feature_dim=64):
        return GlobalLocalFeatureExtractor(
            l=l, H_p=H_p, W_p=W_p, r=r, D=D,
            patch_size=p,
            transformer_cfg=_transformer_cfg(num_heads=4, ff_dim=64),
            feature_dim=feature_dim,
        )

    def test_output_shape(self):
        model = self._make()
        x = torch.rand(2, 12, 8, 8)
        out = model(x)
        assert out.shape == (2, 64)

    def test_feature_map_shape(self):
        model = self._make(l=12, r=4, H_p=8, W_p=8)
        x = torch.rand(2, 12, 8, 8)
        fm = model.extract_feature_map(x)
        # C = ceil(12/4) = 3
        C = math.ceil(12 / 4)
        g_h, g_w = compute_grid_shape(4)
        assert fm.shape == (2, C, g_h * 8, g_w * 8)

    def test_different_feature_dim(self):
        model = self._make(feature_dim=128)
        x = torch.rand(2, 12, 8, 8)
        assert model(x).shape == (2, 128)

    def test_determinism(self):
        model = self._make()
        model.eval()
        x = torch.rand(2, 12, 8, 8)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        assert torch.allclose(o1, o2)

    def test_gradient_flows(self):
        model = self._make()
        x = torch.rand(2, 12, 8, 8)
        out = model(x)
        out.sum().backward()
        # All model parameters should have non-None gradients
        for name, p in model.named_parameters():
            assert p.grad is not None, f"grad is None for: {name}"
            assert p.grad.abs().sum() > 0, f"zero grad for: {name}"

    def test_batch_size_1(self):
        model = self._make()
        x = torch.rand(1, 12, 8, 8)
        out = model(x)
        assert out.shape == (1, 64)

    def test_batch_size_8(self):
        model = self._make()
        x = torch.rand(8, 12, 8, 8)
        out = model(x)
        assert out.shape == (8, 64)

    def test_non_divisible_l(self):
        """l=10, r=4 — should work via zero-padding."""
        model = self._make(l=10, r=4)
        x = torch.rand(2, 10, 8, 8)
        out = model(x)
        assert out.shape[0] == 2

    def test_max_pooling_option(self):
        model = GlobalLocalFeatureExtractor(
            l=12, H_p=8, W_p=8, r=4, D=32, patch_size=2,
            transformer_cfg=_transformer_cfg(num_heads=4),
            feature_dim=64, pooling="max",
        )
        x = torch.rand(2, 12, 8, 8)
        assert model(x).shape == (2, 64)


# ===========================================================================
# MultiViewFeatureExtractor
# ===========================================================================

class TestMultiViewFeatureExtractor:
    def _make(self, feature_dim=64):
        return MultiViewFeatureExtractor(
            l=12,
            ip_shape=(8, 8),
            transport_shape=(8, 8),
            payload_shape=(8, 8),
            r=4, D=32, patch_size=2,
            transformer_cfg=_transformer_cfg(num_heads=4, ff_dim=64),
            feature_dim=feature_dim,
        )

    def test_output_dict_keys(self):
        model = self._make()
        ip  = torch.rand(2, 12, 8, 8)
        tr  = torch.rand(2, 12, 8, 8)
        pay = torch.rand(2, 12, 8, 8)
        out = model(ip, tr, pay)
        assert set(out.keys()) == {"ip_features", "transport_features", "payload_features"}

    def test_output_shapes(self):
        model = self._make(feature_dim=64)
        ip  = torch.rand(2, 12, 8, 8)
        tr  = torch.rand(2, 12, 8, 8)
        pay = torch.rand(2, 12, 8, 8)
        out = model(ip, tr, pay)
        for key in ("ip_features", "transport_features", "payload_features"):
            assert out[key].shape == (2, 64), f"Wrong shape for {key}"

    def test_independent_weights(self):
        """The three extractors must have distinct parameter tensors."""
        model = self._make()
        ip_params  = {id(p) for p in model.ip_extractor.parameters()}
        tr_params  = {id(p) for p in model.transport_extractor.parameters()}
        pay_params = {id(p) for p in model.payload_extractor.parameters()}
        # No parameter shared between any two views
        assert len(ip_params & tr_params)  == 0
        assert len(ip_params & pay_params) == 0
        assert len(tr_params & pay_params) == 0

    def test_from_config(self):
        from preprocessing.view_encoder import ViewEncoder
        config = _make_config(l=12, r=4, D=32, p=2, feature_dim=64, max_bytes=64)
        enc = ViewEncoder(64)
        model = MultiViewFeatureExtractor.from_config(
            config, enc.shape, enc.shape, enc.shape
        )
        ip  = torch.rand(2, 12, *enc.shape)
        tr  = torch.rand(2, 12, *enc.shape)
        pay = torch.rand(2, 12, *enc.shape)
        out = model(ip, tr, pay)
        for key in ("ip_features", "transport_features", "payload_features"):
            assert out[key].shape == (2, 64)

    def test_no_classification_head(self):
        """Must not have any softmax/softplus/linear head for classification."""
        model = self._make()
        # Check output is unbounded (not probability-like)
        ip  = torch.rand(2, 12, 8, 8) * 10
        tr  = torch.rand(2, 12, 8, 8) * 10
        pay = torch.rand(2, 12, 8, 8) * 10
        out = model(ip, tr, pay)
        # If any feature vector sums to ~1.0 it might be a softmax — it shouldn't
        for key in out:
            feat_sum = out[key][0].sum().item()
            assert abs(feat_sum - 1.0) > 0.01, \
                f"{key} looks like a probability distribution (softmax?)"

    def test_gradient_flows_all_views(self):
        model = self._make()
        ip  = torch.rand(2, 12, 8, 8, requires_grad=True)
        tr  = torch.rand(2, 12, 8, 8, requires_grad=True)
        pay = torch.rand(2, 12, 8, 8, requires_grad=True)
        out = model(ip, tr, pay)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert ip.grad is not None  and ip.grad.abs().sum()  > 0
        assert tr.grad is not None  and tr.grad.abs().sum()  > 0
        assert pay.grad is not None and pay.grad.abs().sum() > 0
