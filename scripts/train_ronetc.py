"""
train_ronetc.py

CLI entry point — Phase 4:
    Loads multi-view tensor data → builds RoNeTCClassifier → trains
    using RoNeTCTrainer → saves best model and training history.

Usage
-----
    python scripts/train_ronetc.py

This script expects that `python scripts/run_dataset_adapter.py` (or
the PCAP equivalent) has already been run to produce the `.npy` files.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from data.multiview_dataset import build_multiview_dataloaders
from preprocessing.multiview_preprocessor import load_multiview_arrays
from preprocessing.view_encoder import ViewEncoder
from models.model_factory import build_multiview_extractor
from models.opinion_generator import MultiViewOpinionGenerator
from models.evidence_fusion import DempsterShaferFusion
from models.ronetc_model import RoNeTCClassifier
from models.ronetc_trainer import RoNeTCTrainer
from utils.config import load_config
from utils.paths import get_processed_data_dir, get_models_dir, get_results_dir
from utils.seed import set_seed
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config = load_config()
    set_seed(config["project"]["random_seed"])

    device = torch.device(
        "cuda" if config["device"]["use_cuda_if_available"] and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # --- Load Data ---
    # Try UNSW adapter path first, if it fails, try raw PCAP path
    processed_dir = get_processed_data_dir(config) / "multiview"
    data_source = "unsw"
    source_dir = processed_dir / data_source

    if not (source_dir / "train" / "ip.npy").exists():
        data_source = "raw"
        source_dir = processed_dir / data_source
        if not (source_dir / "train" / "ip.npy").exists():
            logger.error("No multi-view data found. Run Phase 1 ingestion first.")
            sys.exit(1)

    logger.info(f"Loading '{data_source}' multi-view arrays from {source_dir}")
    train_arrays = load_multiview_arrays(source_dir, "train")
    val_arrays = load_multiview_arrays(source_dir, "val")
    # For training we don't strictly need the test arrays here, but we can load them to check
    test_arrays = load_multiview_arrays(source_dir, "test_known")

    batch_size = config["ronetc"]["training"]["batch_size"]
    train_loader, val_loader, test_loader = build_multiview_dataloaders(
        train_arrays, val_arrays, test_arrays,
        batch_size=batch_size
    )

    # --- Build Model ---
    rc = config["ronetc"]
    enc_ip = ViewEncoder(rc["views"]["ip_header"]["max_bytes"])
    enc_tr = ViewEncoder(rc["views"]["transport_header"]["max_bytes"])
    enc_pay = ViewEncoder(rc["views"]["payload"]["max_bytes"])

    extractor = build_multiview_extractor(
        config, enc_ip.shape, enc_tr.shape, enc_pay.shape
    )
    opinion_generator = MultiViewOpinionGenerator.from_config(config)
    fusion = DempsterShaferFusion.from_config(config)
    
    combination_order = rc["fusion"].get("combination_order", ["ip", "transport", "payload"])

    model = RoNeTCClassifier(
        extractor=extractor,
        opinion_generator=opinion_generator,
        fusion=fusion,
        combination_order=combination_order,
    )

    # --- Train ---
    trainer = RoNeTCTrainer(model, device, config)
    models_dir = get_models_dir(config)
    
    logger.info("Starting RoNeTC training...")
    history = trainer.train(train_loader, val_loader, save_dir=models_dir)

    # --- Save Report ---
    results_dir = get_results_dir(config)
    report_path = results_dir / "reports" / "ronetc_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": data_source,
        "device": str(device),
        "epochs_run": len(history["train_loss"]),
        "best_val_loss": min(history["val_loss"]),
        "final_train_loss": history["train_loss"][-1],
        "training_config": rc["training"],
        "history": history,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training complete. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
