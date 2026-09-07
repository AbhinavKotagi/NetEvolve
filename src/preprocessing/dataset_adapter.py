"""
dataset_adapter.py

Purpose: Bridge UNSW-NB15 tabular flow features → RoNeTC multi-view tensor
format, for use when raw PCAP data is not available.

Since UNSW-NB15 provides pre-computed, engineered flow-level statistics
(not raw packets), we cannot extract real IP/transport/payload byte views.
Instead we apply a best-effort mapping:

    IP View       ← IP-layer statistics columns
    Transport View← transport-layer statistics columns
    Payload View  ← payload / connection statistics columns

Column mapping is fully driven by config.yaml (ronetc.dataset_adapter).

Because UNSW-NB15 has one row per flow (not per packet), we replicate the
feature vector `l` times to synthesise `l` identical "packets" per flow.
This is a deliberate approximation documented here and in the README.

Design divergence note
----------------------
The paper's three views are derived from *raw packet bytes* captured with
Wireshark/tcpdump.  This adapter provides a compatibility shim so the
multi-view pipeline can be developed and tested on UNSW-NB15 before real
PCAP data is obtained.  Results on the adapter path will differ from the
paper's results and are not comparable.

RoNeTC Phase 1 — Adapter for UNSW-NB15.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing.view_encoder import ViewEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetAdapter:
    """Convert UNSW-NB15 tabular rows into RoNeTC multi-view tensor format.

    Parameters
    ----------
    config : dict
        Full project config. Uses:
            ronetc.flow.packets_per_flow   (l)
            ronetc.dataset_adapter         (column lists + replicate_as_packets)
            ronetc.views.*max_bytes        (for computing output 2D shapes)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        rc = config["ronetc"]
        self.l = rc["flow"]["packets_per_flow"]
        adapt_cfg = rc["dataset_adapter"]
        self.ip_cols: List[str] = adapt_cfg["ip_columns"]
        self.transport_cols: List[str] = adapt_cfg["transport_columns"]
        self.payload_cols: List[str] = adapt_cfg["payload_columns"]
        self.replicate: bool = adapt_cfg.get("replicate_as_packets", True)

        # Derive output spatial shapes from max_bytes
        self.ip_shape = ViewEncoder.compute_2d_shape(rc["views"]["ip_header"]["max_bytes"])
        self.tr_shape = ViewEncoder.compute_2d_shape(rc["views"]["transport_header"]["max_bytes"])
        self.pay_shape = ViewEncoder.compute_2d_shape(rc["views"]["payload"]["max_bytes"])

    # ------------------------------------------------------------------
    def adapt(
        self,
        df: pd.DataFrame,
        label_col: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Convert a DataFrame of UNSW-NB15 rows to multi-view arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Pre-processed (cleaned, encoded) UNSW-NB15 rows.
        label_col : str, optional
            Column name of the target label. If None, labels are not included.

        Returns
        -------
        dict with keys "ip", "transport", "payload", "labels".
            Shapes: (N, l, H, W) per view, (N,) for labels.
        """
        n = len(df)
        logger.info(
            f"Adapting {n} UNSW-NB15 rows -> multi-view tensors | "
            f"l={self.l} | IP {self.ip_shape}, Transport {self.tr_shape}, "
            f"Payload {self.pay_shape}"
        )

        ip_feat  = self._extract_view(df, self.ip_cols,        self.ip_shape)
        tr_feat  = self._extract_view(df, self.transport_cols, self.tr_shape)
        pay_feat = self._extract_view(df, self.payload_cols,   self.pay_shape)

        # Replicate each row's feature map l times → (N, l, H, W)
        ip_arr  = np.stack([ip_feat]  * self.l, axis=1)  # (N, l, H_ip,  W_ip)
        tr_arr  = np.stack([tr_feat]  * self.l, axis=1)  # (N, l, H_tr,  W_tr)
        pay_arr = np.stack([pay_feat] * self.l, axis=1)  # (N, l, H_pay, W_pay)

        labels = np.array(df[label_col].values, dtype=object) if label_col else None

        return {"ip": ip_arr, "transport": tr_arr, "payload": pay_arr, "labels": labels}

    # ------------------------------------------------------------------
    def _extract_view(
        self,
        df: pd.DataFrame,
        columns: List[str],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Extract and reshape selected columns into a 2D spatial array.

        Steps:
        1. Keep only the configured columns that exist in df (warn if missing).
        2. Min-max scale to [0, 1] per column (independent of train/test split
           here — the adapter is a compatibility shim, not a production scaler;
           production use should fit the scaler only on training data).
        3. Zero-pad or truncate to rows*cols features.
        4. Reshape to (N, rows, cols) float32.

        Parameters
        ----------
        df : pd.DataFrame
        columns : List[str]
            Configured column names for this view.
        shape : Tuple[int, int]
            Target (rows, cols) spatial shape.
        """
        rows, cols = shape
        n_pixels = rows * cols

        # --- Resolve available columns ---
        available = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning(
                f"Columns not found in dataframe (will use zeros): {missing}"
            )

        n = len(df)

        if not available:
            # No columns at all — return zero array
            return np.zeros((n, rows, cols), dtype=np.float32)

        # --- Extract and scale ---
        # Convert string/categorical columns to numeric using factorize
        df_view = df[available].copy()
        for col in df_view.select_dtypes(include=['object', 'category']).columns:
            df_view[col] = pd.factorize(df_view[col])[0]
            
        raw = df_view.values.astype(np.float64)
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled = scaler.fit_transform(raw).astype(np.float32)  # (N, n_feats)

        # --- Pad or truncate to n_pixels features ---
        n_feats = scaled.shape[1]
        if n_feats < n_pixels:
            # Zero-pad on the right
            pad = np.zeros((n, n_pixels - n_feats), dtype=np.float32)
            flat = np.concatenate([scaled, pad], axis=1)  # (N, n_pixels)
        else:
            flat = scaled[:, :n_pixels]                    # (N, n_pixels)

        return flat.reshape(n, rows, cols)
