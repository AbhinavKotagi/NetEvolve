"""
pcap_reader.py

Purpose: Read raw Wireshark .pcap / .pcapng files and extract per-packet
byte fields for each of the three RoNeTC protocol-stack views:
    - IP header bytes
    - Transport layer header bytes (TCP/UDP/ICMP)
    - Payload bytes

Uses scapy for parsing. Malformed or non-IP packets are logged and skipped.
No cleaning, encoding, or normalisation happens here — that is
view_encoder.py's responsibility.

RoNeTC Phase 1 — §3.1 / Section III-A of Wang et al. 2025.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# RawPacket — immutable per-packet record produced by PcapReader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawPacket:
    """One network packet decomposed into its three protocol-stack views.

    All byte fields are raw bytes objects — no normalisation applied here.
    Fields are None when the layer is absent (e.g. UDP has no TCP flags, so
    tcp_flags is 0 and transport_header_bytes is the UDP header).

    Attributes
    ----------
    timestamp : float
        Unix epoch timestamp of the packet capture.
    src_ip : str
        Source IP address (dotted-decimal string).
    dst_ip : str
        Destination IP address.
    src_port : int
        Source port (0 for ICMP).
    dst_port : int
        Destination port (0 for ICMP).
    protocol : str
        Protocol string: "TCP" | "UDP" | "ICMP".
    ip_header_bytes : bytes
        Raw bytes of the IP header.
    transport_header_bytes : bytes
        Raw bytes of the transport-layer header (TCP / UDP / ICMP header).
    payload_bytes : bytes
        Raw bytes of the transport-layer payload (application data). May be
        empty (b"") for packets with no data payload.
    """
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    ip_header_bytes: bytes
    transport_header_bytes: bytes
    payload_bytes: bytes


# ---------------------------------------------------------------------------
# PcapReader
# ---------------------------------------------------------------------------

class PcapReader:
    """Read a .pcap / .pcapng file and return a list of RawPacket objects.

    Only IPv4 packets carrying TCP, UDP, or ICMP are retained. All other
    packets (ARP, IPv6, fragmented IP, etc.) are silently counted and logged
    at the end as a summary.

    Parameters
    ----------
    max_packets : int | None
        If set, stop after reading this many packets from the file. Useful
        for quick smoke-tests on large captures. Default None = read all.
    """

    def __init__(self, max_packets: Optional[int] = None) -> None:
        self.max_packets = max_packets

    # ------------------------------------------------------------------
    def read_pcap(self, path: Path) -> List[RawPacket]:
        """Parse a PCAP/PCAPNG file and return per-packet RawPacket records.

        Parameters
        ----------
        path : Path
            Path to the .pcap or .pcapng file.

        Returns
        -------
        List[RawPacket]
            One entry per accepted packet, in capture order.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PCAP file not found: {path}")

        # Import scapy lazily so the rest of the project is usable even when
        # scapy is not installed (e.g. on machines running only the tabular
        # closed-set pipeline).
        try:
            from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scapy is required for PCAP reading. "
                "Install it with: pip install scapy"
            ) from exc

        logger.info(f"Reading PCAP: {path}")
        try:
            packets_raw = rdpcap(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to read PCAP '{path}': {exc}") from exc

        logger.info(f"Total packets in capture: {len(packets_raw)}")

        results: List[RawPacket] = []
        skipped_non_ip = 0
        skipped_protocol = 0
        skipped_malformed = 0

        for i, pkt in enumerate(packets_raw):
            if self.max_packets is not None and i >= self.max_packets:
                break

            try:
                # --- Must have an IP layer ---
                if not pkt.haslayer(IP):
                    skipped_non_ip += 1
                    continue

                ip_layer = pkt[IP]

                # --- Determine protocol ---
                if pkt.haslayer(TCP):
                    proto = "TCP"
                    transport_layer = pkt[TCP]
                elif pkt.haslayer(UDP):
                    proto = "UDP"
                    transport_layer = pkt[UDP]
                elif pkt.haslayer(ICMP):
                    proto = "ICMP"
                    transport_layer = pkt[ICMP]
                else:
                    skipped_protocol += 1
                    continue

                # --- Extract raw bytes per view ---
                ip_header_bytes = self._extract_ip_header_bytes(ip_layer)
                transport_header_bytes = self._extract_transport_header_bytes(
                    transport_layer, proto
                )
                payload_bytes = self._extract_payload_bytes(pkt, proto)

                # --- Ports (ICMP uses type/code, mapped to 0) ---
                src_port, dst_port = self._extract_ports(transport_layer, proto)

                results.append(RawPacket(
                    timestamp=float(pkt.time),
                    src_ip=ip_layer.src,
                    dst_ip=ip_layer.dst,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=proto,
                    ip_header_bytes=ip_header_bytes,
                    transport_header_bytes=transport_header_bytes,
                    payload_bytes=payload_bytes,
                ))

            except Exception as exc:  # noqa: BLE001
                skipped_malformed += 1
                logger.debug(f"Skipping malformed packet #{i}: {exc}")

        logger.info(
            f"Parsed {len(results)} valid packets | "
            f"skipped: {skipped_non_ip} non-IP, "
            f"{skipped_protocol} unsupported-protocol, "
            f"{skipped_malformed} malformed"
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ip_header_bytes(ip_layer) -> bytes:
        """Return the raw IP header bytes (IHL * 4 bytes, no payload)."""
        # scapy's build() serialises the layer without its payload.
        # bytes(ip_layer) includes the payload — we want header only.
        ihl = ip_layer.ihl if ip_layer.ihl else 5  # default 5 * 4 = 20 bytes
        raw = bytes(ip_layer)
        return raw[: ihl * 4]

    @staticmethod
    def _extract_transport_header_bytes(transport_layer, proto: str) -> bytes:
        """Return the raw transport header bytes without payload."""
        raw = bytes(transport_layer)
        if proto == "TCP":
            # TCP data offset field gives header length in 32-bit words
            data_offset = transport_layer.dataofs if transport_layer.dataofs else 5
            header_len = data_offset * 4
        elif proto == "UDP":
            header_len = 8  # UDP header is always 8 bytes
        else:
            # ICMP: 8 bytes minimum header
            header_len = 8
        return raw[:header_len]

    @staticmethod
    def _extract_payload_bytes(pkt, proto: str) -> bytes:
        """Return raw application-layer payload bytes."""
        try:
            from scapy.all import Raw  # type: ignore
            if pkt.haslayer(Raw):
                return bytes(pkt[Raw].load)
        except Exception:  # noqa: BLE001
            pass
        return b""

    @staticmethod
    def _extract_ports(transport_layer, proto: str):
        """Return (src_port, dst_port) — (0, 0) for ICMP."""
        if proto == "ICMP":
            return 0, 0
        return int(transport_layer.sport), int(transport_layer.dport)
