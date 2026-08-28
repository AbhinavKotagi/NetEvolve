"""
test_loader.py

Tests for src/data/loader.py: missing-file handling and successful loading.
"""
import pandas as pd
import pytest

from data.loader import load_dataset


def test_load_dataset_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset(missing_path)


def test_load_dataset_success(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_csv(csv_path, index=False)

    loaded = load_dataset(csv_path)

    assert loaded.shape == (3, 2)
    assert list(loaded.columns) == ["a", "b"]
    assert loaded["a"].tolist() == [1, 2, 3]
