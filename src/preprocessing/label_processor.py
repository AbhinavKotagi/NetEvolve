"""
label_processor.py

Purpose: Known/unknown class filtering per config/classes.yaml.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def filter_known_unknown(df: pd.DataFrame, classes: dict, target_col: str):
    """Separate known and future-unknown class rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the target column.
    classes : dict
        Mapping with keys ``known_classes`` and ``future_unknown_classes``.
    target_col : str
        Name of the column containing class labels.
    """
    known_classes = set(classes.get("known_classes", []))
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    mask_known = df[target_col].isin(known_classes)
    known_df = df[mask_known].reset_index(drop=True)
    unknown_df = df[~mask_known].reset_index(drop=True)
    return known_df, unknown_df

def save_class_configuration_report(known_df: pd.DataFrame, unknown_df: pd.DataFrame,
                                    classes: dict, target_col: str, report_path: Path):
    """Write a JSON report describing class distribution.

    Parameters
    ----------
    known_df, unknown_df : pd.DataFrame
        Dataframes produced by ``filter_known_unknown``.
    classes : dict
        Original classes mapping.
    target_col : str
        Column name of the label.
    report_path : pathlib.Path
        Destination file path for the JSON report.
    """
    report = {
        "target_column": target_col,
        "known_classes": classes.get("known_classes", []),
        "future_unknown_classes": classes.get("future_unknown_classes", []),
        "known_count": int(len(known_df)),
        "unknown_count": int(len(unknown_df)),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

def fit_label_encoder(known_classes):
    """Fit a LabelEncoder on known classes."""
    encoder = LabelEncoder()
    encoder.fit(known_classes)
    return encoder

def encode_labels(df: pd.DataFrame, target_col: str, encoder: LabelEncoder):
    """Encode target column using fitted encoder.

    Returns a numpy array of integer labels.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    return encoder.transform(df[target_col].values)

def save_label_encoder(encoder: LabelEncoder, models_dir: Path):
    """Persist the LabelEncoder and return a mapping dict.

    Parameters
    ----------
    encoder : sklearn.preprocessing.LabelEncoder
    models_dir : pathlib.Path
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = models_dir / "label_encoder.joblib"
    joblib.dump(encoder, encoder_path)
    mapping = {cls: int(idx) for idx, cls in enumerate(encoder.classes_)}
    return mapping
