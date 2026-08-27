"""
preprocessing_pipeline.py

Purpose: Orchestrates cleaner -> label_processor -> feature_processor end to
end, implementing the exact flow from the project spec:

Original Training CSV -> Filter Known Classes -> Stratified Split -> Train/Val
Original Testing CSV  -> Filter Known Classes -> Final Known-Class Test Set
Future Unknown Classes -> Stored separately, never used for training

Preprocessing objects (scaler/encoder, label encoder) are fit ONLY on the
final training split and reused (never refit) on validation and test data.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# pyrefly: ignore [missing-import]
from src.data.loader import load_training_data, load_testing_data
# pyrefly: ignore [missing-import]
from src.preprocessing.cleaner import clean_data
# pyrefly: ignore [missing-import]
from src.preprocessing.label_processor import (
    filter_known_unknown,
    save_class_configuration_report,
    fit_label_encoder,
    encode_labels,
    save_label_encoder,
)
# pyrefly: ignore [missing-import]
from src.preprocessing.feature_processor import (
    resolve_feature_columns,
    build_preprocessor,
    fit_preprocessor,
    save_preprocessor,
)
# pyrefly: ignore [missing-import]
from src.utils.config import load_config, load_classes
# pyrefly: ignore [missing-import]
from src.utils.paths import get_processed_data_dir, get_models_dir, get_results_dir
# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_preprocessing_pipeline(config: dict = None, classes: dict = None) -> dict:
    config = config or load_config()
    classes = classes or load_classes()

    target_col = config["data"]["target_column"]
    seed = config["project"]["random_seed"]
    val_split = config["training"]["validation_split"]

    processed_dir = get_processed_data_dir(config)
    models_dir = get_models_dir(config)
    results_dir = get_results_dir(config)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- Load + clean raw data ---
    logger.info("=== Loading raw data ===")
    train_raw = load_training_data(config)
    test_raw = load_testing_data(config)

    logger.info("=== Cleaning data ===")
    train_clean = clean_data(train_raw, config)
    test_clean = clean_data(test_raw, config)

    # --- Known / future-unknown filtering ---
    logger.info("=== Filtering known vs. future-unknown classes ===")
    known_train_full, unknown_train = filter_known_unknown(train_clean, classes, target_col)
    known_test, unknown_test = filter_known_unknown(test_clean, classes, target_col)

    save_class_configuration_report(
        known_train_full, unknown_train, classes, target_col,
        results_dir / "reports" / "class_configuration_report.json",
    )

    # --- Stratified train/validation split (from the training CSV only) ---
    logger.info(f"=== Stratified train/validation split ({int((1 - val_split) * 100)}/{int(val_split * 100)}) ===")
    train_df, val_df = train_test_split(
        known_train_full,
        test_size=val_split,
        stratify=known_train_full[target_col],
        random_state=seed,
    )
    logger.info(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows | Test (known): {len(known_test)} rows")

    # --- Feature processing (fit ONLY on train_df) ---
    logger.info("=== Feature processing ===")
    categorical_cols, numerical_cols = resolve_feature_columns(train_df, config)
    feature_cols = categorical_cols + numerical_cols

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    fit_preprocessor(preprocessor, train_df[feature_cols])

    X_train = preprocessor.transform(train_df[feature_cols])
    X_val = preprocessor.transform(val_df[feature_cols])
    X_test = preprocessor.transform(known_test[feature_cols])

    save_preprocessor(preprocessor, models_dir)

    # --- Label encoding (fit on configured known classes, not on any split) ---
    logger.info("=== Label encoding ===")
    encoder = fit_label_encoder(classes["known_classes"])
    y_train = encode_labels(train_df, target_col, encoder)
    y_val = encode_labels(val_df, target_col, encoder)
    y_test = encode_labels(known_test, target_col, encoder)
    class_mapping = save_label_encoder(encoder, models_dir)

    # --- Save processed arrays ---
    logger.info("=== Saving processed data ===")
    np.save(processed_dir / "X_train.npy", X_train)
    np.save(processed_dir / "X_val.npy", X_val)
    np.save(processed_dir / "X_test_known.npy", X_test)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_val.npy", y_val)
    np.save(processed_dir / "y_test_known.npy", y_test)

    metadata = {
        "dataset_source": "UNSW-NB15",
        "preprocessing_date": datetime.now().isoformat(timespec="seconds"),
        "n_samples": {"train": int(len(train_df)), "val": int(len(val_df)), "test_known": int(len(known_test))},
        "n_features": int(X_train.shape[1]),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "class_mapping": class_mapping,
        "random_seed": seed,
    }
    with open(processed_dir / "processed_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"=== Preprocessing complete. Feature matrix width: {X_train.shape[1]} ===")
    return metadata


if __name__ == "__main__":
    run_preprocessing_pipeline()