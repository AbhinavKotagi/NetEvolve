"""
feature_processor.py

Purpose: ColumnTransformer: StandardScaler + OneHotEncoder pipeline.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def resolve_feature_columns(train_df: pd.DataFrame, config: dict):
    """Determine categorical and numerical columns for the pipeline.

    Categorical columns are taken from config['preprocessing']['categorical_columns'].
    Numerical columns are all remaining numeric columns after excluding target and id columns.
    """
    # Load categorical column list from config
    cat_cols = config.get('preprocessing', {}).get('categorical_columns', [])
    # Ensure columns exist in the dataframe
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    # Exclude target and id columns from numeric detection
    target_col = config.get('data', {}).get('target_column')
    id_col = config.get('data', {}).get('id_column')
    exclude = set(cat_cols)
    if target_col:
        exclude.add(target_col)
    if id_col:
        exclude.add(id_col)
    num_cols = [c for c in train_df.select_dtypes(include=['number']).columns if c not in exclude]
    return cat_cols, num_cols

def build_preprocessor(categorical_cols, numerical_cols):
    """Create a ColumnTransformer that scales numeric columns and one-hot encodes categoricals.
    """
    transformers = []
    if numerical_cols:
        transformers.append(('num', StandardScaler(), numerical_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor

def fit_preprocessor(preprocessor, X_train):
    """Fit the ColumnTransformer on training data.
    """
    preprocessor.fit(X_train)
    return preprocessor

def save_preprocessor(preprocessor, models_dir: Path):
    """Persist the fitted preprocessor via joblib.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    preproc_path = models_dir / 'preprocessor.joblib'
    joblib.dump(preprocessor, preproc_path)
    return preproc_path
