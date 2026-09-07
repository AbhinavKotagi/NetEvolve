"""
test_flow_preprocessing.py

Unit tests for Phase 1 components:
    - ViewEncoder.compute_2d_shape()
    - ViewEncoder.encode_view()
    - FlowBuilder (grouping, timeout, bidirectional, min-packet filter)
    - MultiViewFlowDataset shape and optional-label behaviour
    - multiview_preprocessor.preprocess_flows()

All tests use synthetic data — no real PCAP or CSV file required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.flow import Flow, FlowBuilder
from data.pcap_reader import RawPacket
from data.multiview_dataset import MultiViewFlowDataset, build_multiview_dataloaders
from preprocessing.view_encoder import ViewEncoder
from preprocessing.multiview_preprocessor import preprocess_flows


# ===========================================================================
# Helpers
# ===========================================================================

def _make_raw_packet(
    timestamp: float = 0.0,
    src_ip: str = "1.2.3.4",
    dst_ip: str = "5.6.7.8",
    src_port: int = 1234,
    dst_port: int = 80,
    protocol: str = "TCP",
    ip_bytes: bytes = b"\x00" * 20,
    tr_bytes: bytes = b"\x00" * 20,
    pay_bytes: bytes = b"hello world",
) -> RawPacket:
    return RawPacket(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        ip_header_bytes=ip_bytes,
        transport_header_bytes=tr_bytes,
        payload_bytes=pay_bytes,
    )


def _minimal_config(
    packets_per_flow: int = 4,
    timeout: float = 30.0,
    min_pkts: int = 2,
    max_bytes: int = 64,
) -> dict:
    return {
        "ronetc": {
            "flow": {
                "packets_per_flow": packets_per_flow,
                "flow_timeout_seconds": timeout,
                "bidirectional": True,
                "min_packets_per_flow": min_pkts,
            },
            "views": {
                "ip_header":       {"max_bytes": max_bytes},
                "transport_header": {"max_bytes": max_bytes},
                "payload":          {"max_bytes": max_bytes},
            },
        }
    }


# ===========================================================================
# ViewEncoder.compute_2d_shape
# ===========================================================================

class TestCompute2dShape:
    def test_perfect_square(self):
        assert ViewEncoder.compute_2d_shape(64) == (8, 8)

    def test_perfect_square_256(self):
        assert ViewEncoder.compute_2d_shape(256) == (16, 16)

    def test_non_square(self):
        rows, cols = ViewEncoder.compute_2d_shape(128)
        assert rows * cols <= 128
        assert rows >= 1
        assert cols >= rows  # cols >= rows by construction

    def test_single_byte(self):
        rows, cols = ViewEncoder.compute_2d_shape(1)
        assert rows == 1
        assert cols == 1

    def test_product_le_n(self):
        """rows*cols must never exceed n_bytes."""
        for n in [32, 48, 64, 100, 128, 200, 256, 512]:
            rows, cols = ViewEncoder.compute_2d_shape(n)
            assert rows * cols <= n, f"rows*cols={rows*cols} > n={n}"

    def test_deterministic(self):
        for n in [64, 128, 256]:
            assert ViewEncoder.compute_2d_shape(n) == ViewEncoder.compute_2d_shape(n)


# ===========================================================================
# ViewEncoder.encode_view
# ===========================================================================

class TestEncodeView:
    def test_output_shape_exact_fit(self):
        enc = ViewEncoder(64)
        result = enc.encode_view(b"\xff" * 64)
        assert result.shape == enc.shape

    def test_output_dtype(self):
        enc = ViewEncoder(64)
        result = enc.encode_view(b"\x00" * 64)
        assert result.dtype == np.float32

    def test_values_in_range(self):
        enc = ViewEncoder(128)
        result = enc.encode_view(bytes(range(128)))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_all_zeros_when_empty(self):
        enc = ViewEncoder(64)
        result = enc.encode_view(b"")
        assert result.shape == enc.shape
        np.testing.assert_array_equal(result, np.zeros(enc.shape, dtype=np.float32))

    def test_padding_with_short_input(self):
        enc = ViewEncoder(64)
        short = b"\xff" * 10
        result = enc.encode_view(short)
        # First 10 usable pixels should be 1.0
        flat = result.flatten()
        rows, cols = enc.shape
        n_pixels = rows * cols
        # The first min(10, n_pixels) pixels from short are 0xff → 1.0
        n_expected = min(10, n_pixels)
        np.testing.assert_array_almost_equal(flat[:n_expected], np.ones(n_expected))

    def test_truncation_with_long_input(self):
        enc = ViewEncoder(64)
        result = enc.encode_view(b"\xab" * 1000)
        assert result.shape == enc.shape

    def test_max_bytes_255_normalised_to_1(self):
        enc = ViewEncoder(64)
        result = enc.encode_view(b"\xff" * 64)
        np.testing.assert_array_almost_equal(
            result, np.ones(enc.shape, dtype=np.float32)
        )

    def test_zero_bytes_normalised_to_0(self):
        enc = ViewEncoder(64)
        result = enc.encode_view(b"\x00" * 64)
        np.testing.assert_array_almost_equal(
            result, np.zeros(enc.shape, dtype=np.float32)
        )

    def test_different_max_bytes_gives_different_shapes(self):
        enc1 = ViewEncoder(64)
        enc2 = ViewEncoder(128)
        assert enc1.shape != enc2.shape


# ===========================================================================
# FlowBuilder
# ===========================================================================

class TestFlowBuilder:
    def _builder(self, l=4, timeout=30.0, min_pkts=2):
        return FlowBuilder(
            packets_per_flow=l,
            flow_timeout_seconds=timeout,
            min_packets_per_flow=min_pkts,
        )

    def test_groups_packets_by_5_tuple(self):
        pkts = [
            _make_raw_packet(timestamp=float(i), src_port=1234, dst_port=80)
            for i in range(4)
        ]
        flows = self._builder().build_flows(pkts)
        assert len(flows) == 1
        assert flows[0].packet_count == 4

    def test_bidirectional_same_flow(self):
        """A→B and B→A packets should be in the same flow."""
        p_forward  = _make_raw_packet(timestamp=0.0, src_ip="1.1.1.1", dst_ip="2.2.2.2",
                                      src_port=100, dst_port=80)
        p_backward = _make_raw_packet(timestamp=0.5, src_ip="2.2.2.2", dst_ip="1.1.1.1",
                                      src_port=80,  dst_port=100)
        flows = self._builder().build_flows([p_forward, p_backward])
        assert len(flows) == 1

    def test_timeout_creates_new_flow(self):
        pkts = [
            _make_raw_packet(timestamp=0.0),
            _make_raw_packet(timestamp=1.0),
            _make_raw_packet(timestamp=1000.0),  # timeout gap
            _make_raw_packet(timestamp=1001.0),
        ]
        flows = self._builder(timeout=30.0, min_pkts=1).build_flows(pkts)
        assert len(flows) == 2

    def test_different_5_tuples_different_flows(self):
        p1 = _make_raw_packet(src_port=1111, dst_port=80, timestamp=0.0)
        p2 = _make_raw_packet(src_port=2222, dst_port=80, timestamp=0.5)
        p3 = _make_raw_packet(src_port=1111, dst_port=80, timestamp=1.0)
        p4 = _make_raw_packet(src_port=2222, dst_port=80, timestamp=1.5)
        flows = self._builder(min_pkts=1).build_flows([p1, p2, p3, p4])
        assert len(flows) == 2

    def test_truncation_to_l(self):
        pkts = [_make_raw_packet(timestamp=float(i)) for i in range(10)]
        flows = self._builder(l=4, min_pkts=1).build_flows(pkts)
        assert len(flows) == 1
        assert flows[0].packet_count == 4

    def test_min_packets_filter(self):
        """Flows with fewer than min_packets_per_flow are discarded."""
        p = _make_raw_packet(timestamp=0.0)
        flows = self._builder(min_pkts=3).build_flows([p, p])
        assert len(flows) == 0

    def test_empty_input(self):
        flows = self._builder().build_flows([])
        assert flows == []

    def test_flow_id_is_stable(self):
        p = _make_raw_packet(timestamp=0.0)
        flows1 = self._builder(min_pkts=1).build_flows([p])
        flows2 = self._builder(min_pkts=1).build_flows([p])
        assert flows1[0].flow_id == flows2[0].flow_id


# ===========================================================================
# MultiViewFlowDataset
# ===========================================================================

class TestMultiViewFlowDataset:
    def _make_arrays(self, n=10, l=4, h=8, w=8):
        ip = np.random.rand(n, l, h, w).astype(np.float32)
        tr = np.random.rand(n, l, h, w).astype(np.float32)
        pay = np.random.rand(n, l, h, w).astype(np.float32)
        return ip, tr, pay

    def test_len(self):
        ip, tr, pay = self._make_arrays(n=20)
        ds = MultiViewFlowDataset(ip, tr, pay)
        assert len(ds) == 20

    def test_getitem_with_labels(self):
        ip, tr, pay = self._make_arrays(n=5, l=4, h=8, w=8)
        labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        ds = MultiViewFlowDataset(ip, tr, pay, labels=labels)
        ip_v, tr_v, pay_v, lbl = ds[0]
        assert ip_v.shape == (4, 8, 8)
        assert tr_v.shape == (4, 8, 8)
        assert pay_v.shape == (4, 8, 8)
        assert isinstance(lbl, torch.Tensor)

    def test_getitem_without_labels(self):
        ip, tr, pay = self._make_arrays(n=5, l=4, h=8, w=8)
        ds = MultiViewFlowDataset(ip, tr, pay)
        result = ds[0]
        assert len(result) == 3  # no label

    def test_mismatched_lengths_raises(self):
        ip  = np.zeros((10, 4, 8, 8), dtype=np.float32)
        tr  = np.zeros((8,  4, 8, 8), dtype=np.float32)
        pay = np.zeros((10, 4, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            MultiViewFlowDataset(ip, tr, pay)

    def test_view_shapes_property(self):
        ip, tr, pay = self._make_arrays(n=5, l=4, h=8, w=8)
        ds = MultiViewFlowDataset(ip, tr, pay)
        shapes = ds.view_shapes
        assert shapes["ip"] == (4, 8, 8)


# ===========================================================================
# preprocess_flows integration
# ===========================================================================

class TestPreprocessFlows:
    def _make_flows(self, n_flows=5, n_pkts=4):
        flows = []
        for fi in range(n_flows):
            pkts = [
                _make_raw_packet(
                    timestamp=float(fi * 100 + pi),
                    src_port=fi * 100,
                    ip_bytes=bytes([fi, pi] + [0] * 18),
                    pay_bytes=bytes([pi] * 20),
                )
                for pi in range(n_pkts)
            ]
            flow = Flow(
                flow_id=f"flow_{fi:04d}",
                five_tuple=("1.1.1.1", "2.2.2.2", fi * 100, 80, "TCP"),
                packets=pkts,
                start_time=float(fi * 100),
                end_time=float(fi * 100 + n_pkts),
            )
            flows.append(flow)
        return flows

    def test_output_shapes(self):
        config = _minimal_config(packets_per_flow=4, max_bytes=64)
        flows = self._make_flows(n_flows=6, n_pkts=4)
        out = preprocess_flows(flows, config)

        enc = ViewEncoder(64)
        rows, cols = enc.shape

        assert out["ip"].shape       == (6, 4, rows, cols)
        assert out["transport"].shape == (6, 4, rows, cols)
        assert out["payload"].shape   == (6, 4, rows, cols)

    def test_short_flow_zero_padded(self):
        """Flows shorter than l should have zero-padded 'packets' at the end."""
        config = _minimal_config(packets_per_flow=4, max_bytes=64)
        flows = self._make_flows(n_flows=1, n_pkts=2)  # only 2 real packets
        out = preprocess_flows(flows, config)
        ip = out["ip"][0]  # shape (4, rows, cols)
        # Packets 2 and 3 (index 2, 3) should be all zeros (padding)
        assert ip[2].sum() == 0.0
        assert ip[3].sum() == 0.0

    def test_values_in_unit_range(self):
        config = _minimal_config(packets_per_flow=4, max_bytes=64)
        flows = self._make_flows()
        out = preprocess_flows(flows, config)
        for key in ("ip", "transport", "payload"):
            assert out[key].min() >= 0.0
            assert out[key].max() <= 1.0

    def test_with_labels(self):
        config = _minimal_config(packets_per_flow=4, max_bytes=64)
        flows = self._make_flows(n_flows=3)
        labels = ["DoS", "Normal", "Fuzzers"]
        out = preprocess_flows(flows, config, labels=labels)
        assert out["labels"] is not None
        assert list(out["labels"]) == labels

    def test_without_labels(self):
        config = _minimal_config(packets_per_flow=4, max_bytes=64)
        flows = self._make_flows(n_flows=3)
        out = preprocess_flows(flows, config)
        assert out["labels"] is None
