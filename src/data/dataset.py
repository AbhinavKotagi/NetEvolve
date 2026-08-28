"""
dataset.py

Purpose: Custom PyTorch Dataset + DataLoader construction for train/val/test,
built on top of the .npy arrays produced by run_preprocessing.py. This module
performs NO transformation of its own — features are already scaled/encoded.
"""
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NetEvolveDataset(Dataset):
    """Wraps a processed (X, y) pair as tensors. y is optional so this class
    can later be reused for unknown/unlabeled data in Phase 7."""

    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def build_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        NetEvolveDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        NetEvolveDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        NetEvolveDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
