"""
seed.py

Purpose: set_seed() for reproducibility across Python's random module,
NumPy, and PyTorch (including CUDA, if available).
"""
import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch not installed yet is fine for Phase 2 (data-only) scripts.
        pass
