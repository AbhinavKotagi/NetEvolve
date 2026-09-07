"""
run_dataset_adapter.py

CLI entry point — Phase 1 (UNSW-NB15 path):
    UNSW-NB15 CSV → DatasetAdapter → saves per-view .npy arrays

This script is used when raw PCAP data is not yet available.  It applies
the column→view mapping from config.yaml (ronetc.dataset_adapter) to convert
the existing engineered features into the multi-view tensor format required
by the RoNeTC CNN+Transformer architecture (Phase 2 onward).

Saved under:  data/processed/multiview/unsw/<split>/{ip,transport,payload}.npy
Report:       results/reports/multiview_preprocessing_report.json

Usage
-----
    python scripts/run_dataset_adapter.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.loader import load_training_data, load_testing_data
from preprocessing.cleaner import clean_data
from preprocessing.label_processor import (
    filter_known_unknown,
    fit_label_encoder,
    encode_labels,
    save_label_encoder,
)
from preprocessing.dataset_adapter import DatasetAdapter
from preprocessing.multiview_preprocessor import save_multiview_arrays
from utils.config import load_config, load_classes
from utils.paths import get_processed_data_dir, get_results_dir, get_models_dir
from utils.seed import set_seed
from utils.logger import get_logger
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    classes = load_classes()
    set_seed(config["project"]["random_seed"])

    target_col = config["data"]["target_column"]
    seed = config["project"]["random_seed"]
    val_split = config["training"]["validation_split"]

    # --- Load and clean raw CSVs (reuse existing tabular pipeline) ---
    logger.info("Loading raw UNSW-NB15 data...")
    train_raw = load_training_data(config)
    test_raw = load_testing_data(config)

    train_clean = clean_data(train_raw, config)
    test_clean = clean_data(test_raw, config)

    # --- Split known vs unknown (same as tabular pipeline) ---
    known_train_full, unknown_train = filter_known_unknown(train_clean, classes, target_col)
    known_test, unknown_test = filter_known_unknown(test_clean, classes, target_col)

    logger.info(
        f"Known train: {len(known_train_full)} | Known test: {len(known_test)} | "
        f"Unknown train: {len(unknown_train)} | Unknown test: {len(unknown_test)}"
    )

    # --- Stratified train/val split ---
    train_df, val_df = train_test_split(
        known_train_full,
        test_size=val_split,
        stratify=known_train_full[target_col],
        random_state=seed,
    )

    # --- Fit and save label encoder on known classes ---
    models_dir = get_models_dir(config)
    encoder = fit_label_encoder(classes["known_classes"])
    save_label_encoder(encoder, models_dir)

    # --- Adapt each split ---
    adapter = DatasetAdapter(config)

    logger.info("Adapting training split...")
    train_arrays = adapter.adapt(train_df, label_col=target_col)
    train_arrays["labels"] = encode_labels(train_df, target_col, encoder)

    logger.info("Adapting validation split...")
    val_arrays = adapter.adapt(val_df, label_col=target_col)
    val_arrays["labels"] = encode_labels(val_df, target_col, encoder)

    logger.info("Adapting known test split...")
    test_arrays = adapter.adapt(known_test, label_col=target_col)
    test_arrays["labels"] = encode_labels(known_test, target_col, encoder)

    logger.info("Adapting unknown test split (no label encoding — used for open-set eval)...")
    unknown_test_arrays = adapter.adapt(unknown_test, label_col=target_col)
    # Keep string labels for unknown set (not encoded — not in known classes)
    unknown_test_arrays["labels"] = unknown_test[target_col].values

    # --- Save arrays ---
    processed_dir = get_processed_data_dir(config) / "multiview" / "unsw"
    save_multiview_arrays(train_arrays,        processed_dir, split="train")
    save_multiview_arrays(val_arrays,          processed_dir, split="val")
    save_multiview_arrays(test_arrays,         processed_dir, split="test_known")
    save_multiview_arrays(unknown_test_arrays, processed_dir, split="test_unknown")

    # --- Report ---
    results_dir = get_results_dir(config)
    report_path = results_dir / "reports" / "multiview_preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rc = config["ronetc"]
    report = {
        "source": "unsw_nb15_adapter",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test_known": int(len(known_test)),
        "n_test_unknown": int(len(unknown_test)),
        "known_classes": classes["known_classes"],
        "future_unknown_classes": classes["future_unknown_classes"],
        "packets_per_flow": rc["flow"]["packets_per_flow"],
        "ip_columns": rc["dataset_adapter"]["ip_columns"],
        "transport_columns": rc["dataset_adapter"]["transport_columns"],
        "payload_columns": rc["dataset_adapter"]["payload_columns"],
        "output_shapes": {
            "train": {k: list(v.shape) for k, v in train_arrays.items() if v is not None},
            "val": {k: list(v.shape) for k, v in val_arrays.items() if v is not None},
            "test_known": {k: list(v.shape) for k, v in test_arrays.items() if v is not None},
            "test_unknown": {k: list(v.shape) for k, v in unknown_test_arrays.items() if v is not None},
        },
        "divergence_note": (
            "UNSW-NB15 provides engineered flow features, not raw packets. "
            "Each flow row is replicated l times as identical synthetic packets. "
            "Results are not directly comparable to the original RoNeTC paper."
        ),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to: {report_path}")
    logger.info("Dataset adapter complete.")


if __name__ == "__main__":
    main()
