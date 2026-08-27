"""
test_preprocessing.py

Tests for src/preprocessing/feature_processor.py: categorical encoding,
numerical scaling, and that fitting happens only once (no re-fitting on
validation/test data, i.e. no leakage).
"""
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

# pyrefly: ignore [missing-import]
from src.preprocessing.feature_processor import build_preprocessor


@pytest.fixture
def sample_frames():
    train = pd.DataFrame({
        "num_a": [10.0, 20.0, 30.0, 40.0],
        "cat_a": ["x", "y", "x", "y"],
    })
    val = pd.DataFrame({
        "num_a": [1000.0],   # deliberately out-of-range vs. train
        "cat_a": ["z"],      # unseen category
    })
    return train, val


def test_numerical_scaling_fit_on_train_only(sample_frames):
    train, _ = sample_frames
    preprocessor = build_preprocessor(categorical_cols=["cat_a"], numerical_cols=["num_a"])
    transformed = preprocessor.fit_transform(train)

    num_part = transformed[:, :1]  # StandardScaler output is the first column block
    assert np.isclose(num_part.mean(), 0.0, atol=1e-8)
    assert np.isclose(num_part.std(), 1.0, atol=1e-8)


def test_categorical_one_hot_encoding(sample_frames):
    train, _ = sample_frames
    preprocessor = build_preprocessor(categorical_cols=["cat_a"], numerical_cols=["num_a"])
    transformed = preprocessor.fit_transform(train)

    # 1 numerical column + 2 one-hot columns for {x, y}
    assert transformed.shape[1] == 3


def test_no_leakage_transform_does_not_refit(sample_frames):
    """Transforming validation data must use train-fitted statistics, not
    recompute its own mean/std or its own category set."""
    train, val = sample_frames
    preprocessor = build_preprocessor(categorical_cols=["cat_a"], numerical_cols=["num_a"])
    preprocessor.fit(train)

    scaler = preprocessor.named_transformers_["num"]
    train_mean = scaler.mean_[0]

    val_transformed = preprocessor.transform(val)
    # val's raw mean (1000) is wildly different from train's (25) — if the
    # scaler had been refit on val, the scaled value would be ~0, not huge.
    assert not np.isclose(val_transformed[0, 0], 0.0, atol=0.5)
    assert train_mean == pytest.approx(25.0)

    # Unseen category 'z' should be encoded as all-zeros (handle_unknown="ignore"),
    # not raise an error and not add a new column.
    encoder = preprocessor.named_transformers_["cat"]
    assert "z" not in encoder.categories_[0]
    assert val_transformed.shape[1] == 3