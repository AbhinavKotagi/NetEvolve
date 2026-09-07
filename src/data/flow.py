"""
flow.py

Purpose: Group raw packets into bidirectional flows using a 5-tuple key
    (src_ip, dst_ip, src_port, dst_port, protocol).

A flow is bidirectional: packets in both A→B and B→A directions share
one Flow object. The 5-tuple is normalised by sorting (src_ip, dst_ip)
lexicographically so that direction does not create separate flows.

Flow termination rules (applied in order):
    1. Gap between consecutive packets exceeds flow_timeout_seconds.
    2. TCP FIN or RST flag is detected (best-effort — not required).
    3. The capture ends.

Each flow is then truncated to the first `l` packets (packets_per_flow
from config). Flows with fewer than min_packets_per_flow are discarded.

RoNeTC Phase 1 — §3.1 / Section III-A of Wang et al. 2025.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from data.pcap_reader import RawPacket
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Flow — one bidirectional network flow
# ---------------------------------------------------------------------------

@dataclass
class Flow:
    """A bidirectional network flow containing up to `l` packets.

    Attributes
    ----------
    flow_id : str
        SHA-256 hash of (normalised_5-tuple + start_time), truncated to
        16 hex chars. Stable identifier for the flow.
    five_tuple : Tuple[str, str, int, int, str]
        (src_ip_norm, dst_ip_norm, src_port_norm, dst_port_norm, protocol)
        where the IPs/ports are normalised so the lexicographically smaller
        IP is always first.
    packets : List[RawPacket]
        Ordered list of raw packets (max `l` packets).
    start_time : float
        Unix timestamp of the first packet.
    end_time : float
        Unix timestamp of the last packet.
    label : Optional[str]
        Ground-truth class label (populated from the dataset, if available).
    """
    flow_id: str
    five_tuple: Tuple[str, str, int, int, str]
    packets: List[RawPacket] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    label: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def packet_count(self) -> int:
        return len(self.packets)


# ---------------------------------------------------------------------------
# FlowBuilder
# ---------------------------------------------------------------------------

class FlowBuilder:
    """Group a list of RawPackets into bidirectional Flow objects.

    Parameters
    ----------
    packets_per_flow : int
        Maximum number of packets kept per flow (`l` in the paper).
    flow_timeout_seconds : float
        If the gap between two consecutive packets belonging to the same
        5-tuple exceeds this value, a new flow is started.
    min_packets_per_flow : int
        Flows with fewer packets are discarded.
    """

    def __init__(
        self,
        packets_per_flow: int,
        flow_timeout_seconds: float,
        min_packets_per_flow: int,
    ) -> None:
        self.packets_per_flow = packets_per_flow
        self.flow_timeout_seconds = flow_timeout_seconds
        self.min_packets_per_flow = min_packets_per_flow

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: dict) -> "FlowBuilder":
        """Construct from the ronetc.flow section of config.yaml."""
        fc = config["ronetc"]["flow"]
        return cls(
            packets_per_flow=fc["packets_per_flow"],
            flow_timeout_seconds=fc["flow_timeout_seconds"],
            min_packets_per_flow=fc["min_packets_per_flow"],
        )

    # ------------------------------------------------------------------
    def build_flows(self, packets: List[RawPacket]) -> List[Flow]:
        """Group packets into complete, truncated, filtered flows.

        Parameters
        ----------
        packets : List[RawPacket]
            Packets in capture order (sorted by timestamp ascending).

        Returns
        -------
        List[Flow]
            Complete flows with >= min_packets_per_flow packets, each
            containing at most packets_per_flow packets.
        """
        if not packets:
            return []

        # Sort by timestamp to ensure chronological order
        packets = sorted(packets, key=lambda p: p.timestamp)

        # active_flows: normalised_key -> Flow currently being built
        active_flows: Dict[Tuple, Flow] = {}
        # flows that have been finalised (timeout / FIN / RST / capture end)
        completed_flows: List[Flow] = []

        for pkt in packets:
            norm_key = self._normalise_key(pkt)

            if norm_key in active_flows:
                flow = active_flows[norm_key]

                # --- Check for timeout ---
                gap = pkt.timestamp - flow.end_time
                if gap > self.flow_timeout_seconds:
                    # Finalise the current flow and start a new one
                    completed_flows.append(flow)
                    del active_flows[norm_key]
                    flow = self._new_flow(norm_key, pkt)
                    active_flows[norm_key] = flow
                else:
                    # Only append if we still need more packets
                    if len(flow.packets) < self.packets_per_flow:
                        flow.packets.append(pkt)
                    flow.end_time = pkt.timestamp

                    # Best-effort TCP FIN/RST detection
                    if pkt.protocol == "TCP" and self._is_terminating(pkt):
                        completed_flows.append(flow)
                        del active_flows[norm_key]
            else:
                flow = self._new_flow(norm_key, pkt)
                active_flows[norm_key] = flow

        # Finalise all remaining active flows at end-of-capture
        completed_flows.extend(active_flows.values())

        # Filter by minimum packet count
        valid_flows = [
            f for f in completed_flows
            if f.packet_count >= self.min_packets_per_flow
        ]

        logger.info(
            f"Flow building complete: {len(completed_flows)} total flows, "
            f"{len(valid_flows)} valid (>= {self.min_packets_per_flow} packets), "
            f"{len(completed_flows) - len(valid_flows)} discarded (too few packets)"
        )
        return valid_flows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_key(pkt: RawPacket) -> Tuple:
        """Return a normalised 5-tuple that is the same for both directions.

        We sort the (ip, port) pairs so that A→B and B→A map to the same key.
        Design choice: sort by (ip, port) pair lexicographically as a tuple.
        """
        side_a = (pkt.src_ip, pkt.src_port)
        side_b = (pkt.dst_ip, pkt.dst_port)
        if side_a <= side_b:
            return (side_a[0], side_b[0], side_a[1], side_b[1], pkt.protocol)
        return (side_b[0], side_a[0], side_b[1], side_a[1], pkt.protocol)

    @staticmethod
    def _flow_id(norm_key: Tuple, start_time: float) -> str:
        """Stable 16-char hex identifier for this flow."""
        raw = f"{norm_key}|{start_time:.6f}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]

    def _new_flow(self, norm_key: Tuple, first_pkt: RawPacket) -> Flow:
        fid = self._flow_id(norm_key, first_pkt.timestamp)
        return Flow(
            flow_id=fid,
            five_tuple=norm_key,
            packets=[first_pkt],
            start_time=first_pkt.timestamp,
            end_time=first_pkt.timestamp,
        )

    @staticmethod
    def _is_terminating(pkt: RawPacket) -> bool:
        """Heuristic: check if payload bytes hint at a TCP FIN or RST.

        We cannot decode TCP flags here without re-parsing the transport
        header, so we use a minimal check: if transport_header_bytes has
        at least 14 bytes, byte index 13 is the TCP flags byte.
        FIN = 0x01, RST = 0x04.
        """
        hdr = pkt.transport_header_bytes
        if len(hdr) >= 14:
            flags = hdr[13]
            return bool(flags & 0x01) or bool(flags & 0x04)  # FIN or RST
        return False
