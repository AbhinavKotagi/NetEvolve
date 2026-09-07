"""
multiview_preprocessor.py

Purpose: Orchestrate Flow → 3-view 2D tensor encoding for a list of flows.

For each flow:
  - Iterate over its packets (up to `l`).
  - For each packet, encode its IP / transport / payload bytes using
    ViewEncoder → (rows, cols) float32 array.
  - Stack packets per view → (l, rows, cols) per flow.
  - Flows with fewer than `l` packets are zero-padded with extra "packets".

Final output per split:
    ip_arrays        : np.ndarray  shape (N, l, H_ip,  W_ip)
    transport_arrays : np.ndarray  shape (N, l, H_tr,  W_tr)
    payload_arrays   : np.ndarray  shape (N, l, H_pay, W_pay)
    labels           : np.ndarray  shape (N,)  dtype str | None

RoNeTC Phase 1 — §3.1 / Section III-A of Wang et al. 2025.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from data.flow import Flow
from preprocessing.view_encoder import ViewEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_flows(
    flows: List[Flow],
    config: dict,
    labels: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Encode a list of flows into multi-view numpy arrays.

    Parameters
    ----------
    flows : List[Flow]
        Flows produced by FlowBuilder.  Each flow has packets in
        capture order, already truncated to packets_per_flow.
    config : dict
        Full project config (ronetc section is used).
    labels : list of str, optional
        Per-flow ground-truth labels (same length as *flows*).
        If None, the returned "labels" key will be None.

    Returns
    -------
    dict with keys:
        "ip"        : np.ndarray  (N, l, H_ip,  W_ip)   float32
        "transport" : np.ndarray  (N, l, H_tr,  W_tr)   float32
        "payload"   : np.ndarray  (N, l, H_pay, W_pay)  float32
        "labels"    : np.ndarray  (N,) of str, or None
    """
    rc = config["ronetc"]
    l_max = rc["flow"]["packets_per_flow"]

    # --- Build view encoders ---
    enc_ip  = ViewEncoder(rc["views"]["ip_header"]["max_bytes"])
    enc_tr  = ViewEncoder(rc["views"]["transport_header"]["max_bytes"])
    enc_pay = ViewEncoder(rc["views"]["payload"]["max_bytes"])

    H_ip,  W_ip  = enc_ip.shape
    H_tr,  W_tr  = enc_tr.shape
    H_pay, W_pay = enc_pay.shape

    n_flows = len(flows)
    logger.info(
        f"Encoding {n_flows} flows | l={l_max} packets | "
        f"IP view {H_ip}×{W_ip}, Transport view {H_tr}×{W_tr}, "
        f"Payload view {H_pay}×{W_pay}"
    )

    # Pre-allocate output arrays (zero = padding value for short flows)
    ip_arr  = np.zeros((n_flows, l_max, H_ip,  W_ip),  dtype=np.float32)
    tr_arr  = np.zeros((n_flows, l_max, H_tr,  W_tr),  dtype=np.float32)
    pay_arr = np.zeros((n_flows, l_max, H_pay, W_pay), dtype=np.float32)

    for fi, flow in enumerate(flows):
        for pi, pkt in enumerate(flow.packets[:l_max]):
            ip_arr[fi, pi]  = enc_ip.encode_view(pkt.ip_header_bytes)
            tr_arr[fi, pi]  = enc_tr.encode_view(pkt.transport_header_bytes)
            pay_arr[fi, pi] = enc_pay.encode_view(pkt.payload_bytes)

    out: Dict[str, np.ndarray] = {
        "ip": ip_arr,
        "transport": tr_arr,
        "payload": pay_arr,
    }

    if labels is not None:
        out["labels"] = np.array(labels, dtype=object)
    else:
        out["labels"] = None  # type: ignore[assignment]

    logger.info("Multi-view encoding complete.")
    return out


def save_multiview_arrays(arrays: Dict, save_dir: Path, split: str) -> None:
    """Persist encoded arrays to disk under save_dir/<split>/.

    Parameters
    ----------
    arrays : dict
        Output of preprocess_flows().
    save_dir : Path
        Root directory (e.g. data/processed/multiview/).
    split : str
        One of "train", "val", "test".
    """
    split_dir = save_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    for view in ("ip", "transport", "payload"):
        path = split_dir / f"{view}.npy"
        np.save(path, arrays[view])
        logger.info(f"Saved {view} view → {path}  shape={arrays[view].shape}")

    if arrays.get("labels") is not None:
        lbl_path = split_dir / "labels.npy"
        np.save(lbl_path, arrays["labels"])
        logger.info(f"Saved labels → {lbl_path}")


def load_multiview_arrays(save_dir: Path, split: str) -> Dict[str, np.ndarray]:
    """Load previously saved multi-view arrays from disk."""
    split_dir = save_dir / split
    out: Dict[str, np.ndarray] = {}
    for view in ("ip", "transport", "payload"):
        path = split_dir / f"{view}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {view} array for split '{split}': {path}. "
                "Run preprocessing first."
            )
        out[view] = np.load(path)
    lbl_path = split_dir / "labels.npy"
    out["labels"] = np.load(lbl_path, allow_pickle=True) if lbl_path.exists() else None  # type: ignore
    return out
