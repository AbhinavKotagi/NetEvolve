"""
multiview_dataset.py

Purpose: PyTorch Dataset + DataLoader wrappers for the RoNeTC multi-view
tensor format produced by multiview_preprocessor.py or dataset_adapter.py.

Each sample is a triple of 2D tensors (one per protocol-stack view) plus
an optional integer label.  Labels are optional so the same class can be
used for labelled training data AND for unlabelled unknown-class data
during open-set discovery (Phase 6).

RoNeTC Phase 1 — §3.1 / Section III-A of Wang et al. 2025.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.logger import get_logger

logger = get_logger(__name__)


class MultiViewFlowDataset(Dataset):
    """Dataset wrapping three per-view numpy arrays (and optional labels).

    Parameters
    ----------
    ip_tensors : np.ndarray
        Shape (N, l, H_ip, W_ip) — IP header view.
    transport_tensors : np.ndarray
        Shape (N, l, H_tr, W_tr) — Transport header view.
    payload_tensors : np.ndarray
        Shape (N, l, H_pay, W_pay) — Payload view.
    labels : np.ndarray or None
        Shape (N,) integer class indices.  If None, __getitem__ returns
        only the three view tensors (useful for unlabelled inference).
    """

    def __init__(
        self,
        ip_tensors: np.ndarray,
        transport_tensors: np.ndarray,
        payload_tensors: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        n = ip_tensors.shape[0]
        if transport_tensors.shape[0] != n or payload_tensors.shape[0] != n:
            raise ValueError(
                "All three view arrays must have the same number of samples "
                f"(got ip={n}, transport={transport_tensors.shape[0]}, "
                f"payload={payload_tensors.shape[0]})"
            )
        if labels is not None and len(labels) != n:
            raise ValueError(
                f"labels length {len(labels)} does not match n_samples {n}"
            )

        self.ip = torch.as_tensor(ip_tensors, dtype=torch.float32)
        self.transport = torch.as_tensor(transport_tensors, dtype=torch.float32)
        self.payload = torch.as_tensor(payload_tensors, dtype=torch.float32)
        self.labels: Optional[torch.Tensor] = (
            torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.ip.shape[0]

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        ip_view  = self.ip[idx]           # (l, H_ip,  W_ip)
        tr_view  = self.transport[idx]    # (l, H_tr,  W_tr)
        pay_view = self.payload[idx]      # (l, H_pay, W_pay)

        if self.labels is not None:
            return ip_view, tr_view, pay_view, self.labels[idx]
        return ip_view, tr_view, pay_view

    # ------------------------------------------------------------------
    @property
    def view_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return the per-view tensor shapes (excluding batch dimension)."""
        return {
            "ip": tuple(self.ip.shape[1:]),
            "transport": tuple(self.transport.shape[1:]),
            "payload": tuple(self.payload.shape[1:]),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_multiview_dataloaders(
    train_arrays: Dict[str, np.ndarray],
    val_arrays: Dict[str, np.ndarray],
    test_arrays: Dict[str, np.ndarray],
    label_encoder=None,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from multi-view array dicts.

    Parameters
    ----------
    train_arrays, val_arrays, test_arrays : dict
        Output of preprocess_flows() or DatasetAdapter.adapt() with keys
        "ip", "transport", "payload", "labels".
    label_encoder : sklearn LabelEncoder, optional
        If provided, encodes string labels to integers.  If labels are
        already integers, leave as None.
    batch_size : int

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """

    def _make_labels(arrays: Dict, le) -> Optional[np.ndarray]:
        lbl = arrays.get("labels")
        if lbl is None:
            return None
        if le is not None:
            lbl = le.transform(lbl)
        return np.array(lbl, dtype=np.int64)

    train_lbl = _make_labels(train_arrays, label_encoder)
    val_lbl   = _make_labels(val_arrays,   label_encoder)
    test_lbl  = _make_labels(test_arrays,  label_encoder)

    train_ds = MultiViewFlowDataset(
        train_arrays["ip"], train_arrays["transport"], train_arrays["payload"],
        labels=train_lbl,
    )
    val_ds = MultiViewFlowDataset(
        val_arrays["ip"], val_arrays["transport"], val_arrays["payload"],
        labels=val_lbl,
    )
    test_ds = MultiViewFlowDataset(
        test_arrays["ip"], test_arrays["transport"], test_arrays["payload"],
        labels=test_lbl,
    )

    logger.info(
        f"DataLoaders: train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)} | batch_size={batch_size}"
    )
    logger.info(f"View shapes: {train_ds.view_shapes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
