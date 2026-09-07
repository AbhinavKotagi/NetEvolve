"""
view_encoder.py

Purpose: Convert a raw bytes object (one protocol-stack view for one packet)
into a fixed-size, normalised 2D float32 numpy array.

Steps
-----
1. Truncate raw bytes to max_bytes, or zero-pad if shorter (0x00 padding).
2. Compute a near-square (rows, cols) shape for max_bytes.
3. Reshape the 1D byte array into (rows, cols).
4. Normalise to [0.0, 1.0] by dividing by 255.

Design choices (underspecified in the paper)
--------------------------------------------
- 2D shape: rows = floor(sqrt(max_bytes)), cols = max_bytes // rows.
  If rows * cols < max_bytes (due to integer division), the remainder bytes
  are dropped (documented; deterministic).  The paper does not specify an
  exact reshaping convention.
- Padding value: 0x00. Explicitly distinct from real data so a model
  can learn to ignore it.
- Normalisation: divide by 255 (unsigned byte range). Not z-score, because
  byte values have a natural [0, 255] range.

RoNeTC Phase 1 — §3.1 / Section III-A of Wang et al. 2025.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ViewEncoder:
    """Encode a single packet's byte view into a fixed-size 2D float32 array.

    Parameters
    ----------
    max_bytes : int
        Maximum bytes kept per view per packet.  Determines the output shape.
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self.max_bytes = max_bytes
        self.shape: Tuple[int, int] = self.compute_2d_shape(max_bytes)

    # ------------------------------------------------------------------
    @staticmethod
    def compute_2d_shape(n_bytes: int) -> Tuple[int, int]:
        """Compute a near-square (rows, cols) grid for n_bytes pixels.

        Algorithm
        ---------
        rows = floor(sqrt(n_bytes))
        cols = n_bytes // rows

        rows * cols <= n_bytes always holds. Any remainder bytes
        (n_bytes - rows * cols) are intentionally dropped to keep the
        shape deterministic without padding the 2D grid itself.

        Examples
        --------
        >>> ViewEncoder.compute_2d_shape(128)
        (11, 11)   # 11 * 11 = 121, 7 bytes dropped
        >>> ViewEncoder.compute_2d_shape(64)
        (8, 8)     # exact square
        >>> ViewEncoder.compute_2d_shape(256)
        (16, 16)   # exact square

        Design note: we choose rows <= cols (since floor(sqrt) <=
        ceil(sqrt)), giving a landscape-oriented or square grid.
        """
        if n_bytes < 1:
            raise ValueError(f"n_bytes must be >= 1, got {n_bytes}")
        rows = int(math.isqrt(n_bytes))  # integer square root (Python 3.8+)
        if rows == 0:
            rows = 1
        cols = n_bytes // rows
        return rows, cols

    # ------------------------------------------------------------------
    def encode_view(self, raw_bytes: bytes) -> np.ndarray:
        """Encode raw bytes into a (rows, cols) float32 array in [0, 1].

        Parameters
        ----------
        raw_bytes : bytes
            Raw bytes for one view of one packet. Length may be anything.

        Returns
        -------
        np.ndarray
            Shape (rows, cols), dtype float32, values in [0.0, 1.0].
        """
        rows, cols = self.shape
        n_pixels = rows * cols

        # --- Step 1: truncate or zero-pad to max_bytes ---
        buf = bytearray(self.max_bytes)   # initialised to 0x00
        n_copy = min(len(raw_bytes), self.max_bytes)
        buf[:n_copy] = raw_bytes[:n_copy]

        # --- Step 2: keep only the first n_pixels bytes (may trim remainder) ---
        pixel_bytes = bytes(buf[:n_pixels])

        # --- Step 3: reshape to 2D ---
        arr = np.frombuffer(pixel_bytes, dtype=np.uint8).reshape(rows, cols)

        # --- Step 4: normalise ---
        return arr.astype(np.float32) / 255.0
