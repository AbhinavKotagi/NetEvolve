"""
run_pcap_ingestion.py

CLI entry point — Phase 1 (PCAP path):
    PCAP file(s) → PcapReader → FlowBuilder → multiview_preprocessor
    → saves per-view .npy arrays to data/processed/multiview/

Usage
-----
    python scripts/run_pcap_ingestion.py --input data/raw/pcaps/sample.pcap
    python scripts/run_pcap_ingestion.py --input data/raw/pcaps/  # whole dir
    python scripts/run_pcap_ingestion.py --input sample.pcap --max-packets 5000

The script saves:
    data/processed/multiview/raw/<split>/{ip,transport,payload}.npy
    data/processed/multiview/raw/<split>/labels.npy  (if labels provided)
    results/reports/multiview_preprocessing_report.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.pcap_reader import PcapReader
from data.flow import FlowBuilder
from preprocessing.multiview_preprocessor import (
    preprocess_flows,
    save_multiview_arrays,
)
from utils.config import load_config
from utils.paths import get_processed_data_dir, get_results_dir
from utils.seed import set_seed
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RoNeTC Phase 1: PCAP → multi-view tensor arrays"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a .pcap/.pcapng file or directory of PCAP files"
    )
    parser.add_argument(
        "--split", default="test",
        help="Split name for saving (default: test). Use 'train'/'val'/'test'."
    )
    parser.add_argument(
        "--max-packets", type=int, default=None,
        help="Limit packets read per PCAP file (useful for quick testing)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    set_seed(config["project"]["random_seed"])

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # --- Collect PCAP files ---
    if input_path.is_dir():
        pcap_files = sorted(input_path.glob("*.pcap")) + sorted(input_path.glob("*.pcapng"))
    else:
        pcap_files = [input_path]

    if not pcap_files:
        logger.error(f"No .pcap/.pcapng files found in: {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(pcap_files)} PCAP file(s)")

    # --- Parse all packets ---
    reader = PcapReader(max_packets=args.max_packets)
    all_packets = []
    for pf in pcap_files:
        pkts = reader.read_pcap(pf)
        all_packets.extend(pkts)
    logger.info(f"Total packets parsed: {len(all_packets)}")

    # --- Build flows ---
    builder = FlowBuilder.from_config(config)
    flows = builder.build_flows(all_packets)
    logger.info(f"Total valid flows: {len(flows)}")

    if not flows:
        logger.error("No valid flows produced. Check min_packets_per_flow setting.")
        sys.exit(1)

    # --- Encode multi-view ---
    arrays = preprocess_flows(flows, config)

    # --- Save arrays ---
    processed_dir = get_processed_data_dir(config) / "multiview" / "raw"
    save_multiview_arrays(arrays, processed_dir, split=args.split)

    # --- Save report ---
    results_dir = get_results_dir(config)
    report_path = results_dir / "reports" / "multiview_preprocessing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rc = config["ronetc"]
    from preprocessing.view_encoder import ViewEncoder
    report = {
        "source": "pcap",
        "pcap_files": [str(f) for f in pcap_files],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_packets": len(all_packets),
        "total_flows": len(flows),
        "split": args.split,
        "packets_per_flow": rc["flow"]["packets_per_flow"],
        "ip_view_shape": list(ViewEncoder.compute_2d_shape(
            rc["views"]["ip_header"]["max_bytes"])),
        "transport_view_shape": list(ViewEncoder.compute_2d_shape(
            rc["views"]["transport_header"]["max_bytes"])),
        "payload_view_shape": list(ViewEncoder.compute_2d_shape(
            rc["views"]["payload"]["max_bytes"])),
        "output_shapes": {k: list(v.shape) for k, v in arrays.items() if v is not None},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
