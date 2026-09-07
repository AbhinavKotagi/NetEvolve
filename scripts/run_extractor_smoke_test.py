"""
run_extractor_smoke_test.py

CLI smoke test — Phase 2:
    Build a batch of synthetic multi-view samples and run them through
    the MultiViewFeatureExtractor. Reports output shapes, parameter count,
    and forward-pass timing.

Saves report to: results/reports/extractor_smoke_test_report.json

Usage
-----
    python scripts/run_extractor_smoke_test.py
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from models.model_factory import build_multiview_extractor
from preprocessing.view_encoder import ViewEncoder
from utils.config import load_config
from utils.paths import get_results_dir
from utils.seed import set_seed
from utils.logger import get_logger

logger = get_logger(__name__)


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main() -> None:
    config = load_config()
    set_seed(config["project"]["random_seed"])

    rc = config["ronetc"]
    l = rc["flow"]["packets_per_flow"]
    batch_size = 8  # synthetic batch

    # --- Compute view shapes ---
    enc_ip  = ViewEncoder(rc["views"]["ip_header"]["max_bytes"])
    enc_tr  = ViewEncoder(rc["views"]["transport_header"]["max_bytes"])
    enc_pay = ViewEncoder(rc["views"]["payload"]["max_bytes"])

    ip_shape  = enc_ip.shape
    tr_shape  = enc_tr.shape
    pay_shape = enc_pay.shape

    logger.info(
        f"View shapes — IP: {ip_shape}, Transport: {tr_shape}, Payload: {pay_shape}"
    )
    logger.info(f"l={l}, batch_size={batch_size}")

    # --- Build model ---
    model = build_multiview_extractor(config, ip_shape, tr_shape, pay_shape)
    model.eval()

    params = count_parameters(model)
    logger.info(
        f"Model parameters: {params['trainable']:,} trainable / {params['total']:,} total"
    )

    # --- Synthetic input ---
    ip  = torch.rand(batch_size, l, *ip_shape)
    tr  = torch.rand(batch_size, l, *tr_shape)
    pay = torch.rand(batch_size, l, *pay_shape)

    # --- Forward pass timing ---
    N_WARMUP = 2
    N_TIMED = 10
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(ip, tr, pay)
        t0 = time.perf_counter()
        for _ in range(N_TIMED):
            out = model(ip, tr, pay)
        elapsed = (time.perf_counter() - t0) / N_TIMED * 1000  # ms per forward pass

    # --- Log output shapes ---
    for key, feat in out.items():
        logger.info(f"  {key}: {tuple(feat.shape)}")

    logger.info(f"Forward-pass time: {elapsed:.2f} ms (avg over {N_TIMED} runs)")

    # --- Verify shapes ---
    feature_dim = rc["global_local_extractor"]["feature_dim"]
    for key, feat in out.items():
        assert feat.shape == (batch_size, feature_dim), \
            f"Expected ({batch_size}, {feature_dim}) for {key}, got {feat.shape}"

    # --- Save report ---
    results_dir = get_results_dir(config)
    report_path = results_dir / "reports" / "extractor_smoke_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    gle_cfg = rc["global_local_extractor"]
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_size": batch_size,
        "packets_per_flow": l,
        "view_shapes": {
            "ip": list(ip_shape),
            "transport": list(tr_shape),
            "payload": list(pay_shape),
        },
        "extractor_config": {
            "packets_per_channel": gle_cfg["packets_per_channel"],
            "patch_size": gle_cfg["patch_size"],
            "local_conv_channels": gle_cfg["local_conv_channels"],
            "transformer_layers": gle_cfg["transformer"]["num_layers"],
            "transformer_heads": gle_cfg["transformer"]["num_heads"],
            "feature_dim": gle_cfg["feature_dim"],
            "pooling": gle_cfg["pooling"],
        },
        "model_parameters": params,
        "output_shapes": {k: list(v.shape) for k, v in out.items()},
        "forward_pass_ms_avg": round(elapsed, 2),
        "status": "PASS",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Smoke test PASSED. Report saved to: {report_path}")


if __name__ == "__main__":
    main()
