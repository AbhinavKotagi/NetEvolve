"""
cleaner.py

Purpose: Missing value handling, duplicate removal, infinite value handling.
"""

import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Basic data cleaning:
    - Drop duplicate rows
    - Fill numeric NaNs with median
    - Fill categorical NaNs with mode
    - Replace infinite values with NaN then fill as above
    """
    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    # Fill numeric NaNs with median
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    # Fill categorical NaNs with mode (most frequent)
    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
    return df
