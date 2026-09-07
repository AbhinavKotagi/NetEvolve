"""
test_ronetc_model.py

Unit tests for the unified RoNeTC PyTorch model.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.ronetc_model import RoNeTCClassifier


def _make_config(
    l=12, r=4, D=32, p=2, feature_dim=64, num_classes=5, max_bytes=64
):
    return {
        "ronetc": {
            "flow": {"packets_per_flow": l},
            "views": {
                "ip_header": {"max_bytes": max_bytes},
                "transport_header": {"max_bytes": max_bytes},
                "payload": {"max_bytes": max_bytes},
            },
            "global_local_extractor": {
                "enabled": True,
                "packets_per_channel": r,
                "patch_size": p,
                "local_conv_channels": D,
                "transformer": {
                    "num_layers": 2,
                    "num_heads": 4,
                    "ff_dim": 64,
                    "dropout": 0.0,
                },
                "pooling": "avg",
                "feature_dim": feature_dim,
            },
            "opinion_generator": {
                "num_classes": num_classes,
            },
            "fusion": {
                "combination_order": ["ip", "transport", "payload"],
                "conflict_clamp_min": 1e-7,
            },
        }
    }


class TestRoNeTCClassifier:
    def test_forward_returns_dict_of_opinions(self):
        from preprocessing.view_encoder import ViewEncoder
        config = _make_config(l=12, r=4, D=32, p=2, feature_dim=64, num_classes=5)
        enc = ViewEncoder(64)
        shape = enc.shape

        model = RoNeTCClassifier.from_config(config, shape, shape, shape)

        B = 2
        ip = torch.rand(B, 12, *shape)
        tr = torch.rand(B, 12, *shape)
        pay = torch.rand(B, 12, *shape)

        out = model(ip, tr, pay)

        # Expected keys: "ip", "transport", "payload", "fused"
        assert set(out.keys()) == {"ip", "transport", "payload", "fused"}

        for view in ["ip", "transport", "payload", "fused"]:
            view_out = out[view]
            assert "evidence" in view_out
            assert "alpha" in view_out
            assert "belief" in view_out
            assert "uncertainty" in view_out
            assert "S" in view_out

            assert view_out["evidence"].shape == (B, 5)

    def test_invalid_combination_order_raises(self):
        from preprocessing.view_encoder import ViewEncoder
        config = _make_config(l=12, r=4, D=32, p=2, feature_dim=64, num_classes=5)
        config["ronetc"]["fusion"]["combination_order"] = ["ip", "magic", "payload"]
        enc = ViewEncoder(64)
        shape = enc.shape

        with pytest.raises(ValueError, match="Invalid view"):
            RoNeTCClassifier.from_config(config, shape, shape, shape)

    def test_gradient_flows(self):
        from preprocessing.view_encoder import ViewEncoder
        config = _make_config(l=12, r=4, D=32, p=2, feature_dim=64, num_classes=5)
        enc = ViewEncoder(64)
        shape = enc.shape

        model = RoNeTCClassifier.from_config(config, shape, shape, shape)

        B = 2
        ip = torch.rand(B, 12, *shape, requires_grad=True)
        tr = torch.rand(B, 12, *shape, requires_grad=True)
        pay = torch.rand(B, 12, *shape, requires_grad=True)

        out = model(ip, tr, pay)
        fused_evidence = out["fused"]["evidence"]
        
        loss = fused_evidence.sum()
        loss.backward()

        assert ip.grad is not None
        assert tr.grad is not None
        assert pay.grad is not None
        assert ip.grad.abs().sum() > 0
