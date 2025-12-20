"""
Memory Coalescing Unit for SIMT Streaming Multiprocessor.

Analyzes warp memory access patterns and combines adjacent accesses into
efficient 128-byte memory transactions.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   MEMORY COALESCING UNIT                             │
    │                                                                      │
    │  Input: 32 thread addresses (one warp)                              │
    │  Output: N memory transactions (128-byte aligned segments)           │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │                  SEGMENT ANALYSIS                             │   │
    │  │                                                               │   │
    │  │  Thread 0:  addr=0x1000 ──┐                                  │   │
    │  │  Thread 1:  addr=0x1004 ──┤                                  │   │
    │  │  Thread 2:  addr=0x1008 ──┼── Segment 0x1000 (1 transaction) │   │
    │  │  Thread 3:  addr=0x100C ──┤                                  │   │
    │  │  ...        ...        ──┘                                   │   │
    │  │  Thread 31: addr=0x107C ──┘                                  │   │
    │  │                                                               │   │
    │  │  Coalesced: 32 accesses → 1 transaction (ideal case)         │   │
    │  │  Strided:   32 accesses → N transactions (N segments hit)    │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    │  Access Patterns:                                                    │
    │    • Coalesced (stride=4):    1 transaction  (ideal)                │
    │    • Strided (stride=128):   32 transactions (worst)                │
    │    • Random:                 varies by pattern                       │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import SIMTConfig


@dataclass
class CoalescingResult:
    """Result of coalescing analysis for a warp's memory access."""

    # Number of 128-byte transactions needed
    num_transactions: int = 0

    # Segments accessed (128-byte aligned base addresses)
    segments: list[int] = field(default_factory=list)

    # Per-segment thread masks (which threads access each segment)
    segment_thread_masks: dict[int, int] = field(default_factory=dict)

    # Per-segment addresses within the segment
    segment_addresses: dict[int, list[tuple[int, int]]] = field(
        default_factory=dict
    )  # segment -> [(thread_id, offset), ...]

    # Efficiency metrics
    active_threads: int = 0
    bytes_accessed: int = 0  # Actual data needed
    bytes_transferred: int = 0  # Data transferred (transactions * 128)

    @property
    def efficiency(self) -> float:
        """Coalescing efficiency (0.0 to 1.0)."""
        if self.bytes_transferred == 0:
            return 0.0
        return self.bytes_accessed / self.bytes_transferred

    @property
    def is_fully_coalesced(self) -> bool:
        """True if all accesses fit in one transaction."""
        return self.num_transactions == 1


@dataclass
class MemoryCoalescerSim:
    """
    Behavioral simulation model for memory coalescing.

    Analyzes 32 thread addresses and determines how many 128-byte
    memory transactions are needed.
    """

    config: SIMTConfig

    # Statistics
    total_analyses: int = 0
    total_transactions: int = 0
    total_threads_served: int = 0
    total_bytes_accessed: int = 0
    total_bytes_transferred: int = 0
    total_energy_pj: float = 0.0

    def analyze(
        self,
        addresses: Sequence[int | None],
        thread_mask: int = 0xFFFF_FFFF,
        access_size: int = 4,
    ) -> CoalescingResult:
        """
        Analyze a warp's memory access pattern for coalescing.

        Args:
            addresses: List of 32 addresses (one per thread), None if inactive
            thread_mask: Bitmask of active threads
            access_size: Bytes per thread access (default 4 for 32-bit)

        Returns:
            CoalescingResult with transaction analysis
        """
        result = CoalescingResult()
        segment_size = self.config.coalescing_window  # 128 bytes

        # Analyze each active thread's address
        for thread_id in range(32):
            if not ((thread_mask >> thread_id) & 1):
                continue
            addr = addresses[thread_id] if thread_id < len(addresses) else None
            if addr is None:
                continue

            # Compute 128-byte aligned segment
            segment_base = (addr // segment_size) * segment_size
            offset_in_segment = addr % segment_size

            # Track segment access
            if segment_base not in result.segments:
                result.segments.append(segment_base)
                result.segment_thread_masks[segment_base] = 0
                result.segment_addresses[segment_base] = []

            result.segment_thread_masks[segment_base] |= 1 << thread_id
            result.segment_addresses[segment_base].append((thread_id, offset_in_segment))
            result.active_threads += 1
            result.bytes_accessed += access_size

        # Sort segments for deterministic ordering
        result.segments.sort()
        result.num_transactions = len(result.segments)
        result.bytes_transferred = result.num_transactions * segment_size

        # Update statistics
        self.total_analyses += 1
        self.total_transactions += result.num_transactions
        self.total_threads_served += result.active_threads
        self.total_bytes_accessed += result.bytes_accessed
        self.total_bytes_transferred += result.bytes_transferred
        self.total_energy_pj += result.num_transactions * self.config.coalescing_energy_pj

        return result

    def analyze_strided(
        self,
        base_addr: int,
        stride: int,
        thread_mask: int = 0xFFFF_FFFF,
    ) -> CoalescingResult:
        """
        Analyze strided access pattern (common for matrix row access).

        Args:
            base_addr: Starting address for thread 0
            stride: Address stride between consecutive threads
            thread_mask: Bitmask of active threads

        Returns:
            CoalescingResult with transaction analysis
        """
        addresses = [base_addr + tid * stride for tid in range(32)]
        return self.analyze(addresses, thread_mask)

    def get_statistics(self) -> dict[str, Any]:
        """Get coalescing statistics."""
        avg_transactions = self.total_transactions / max(1, self.total_analyses)
        avg_efficiency = self.total_bytes_accessed / max(1, self.total_bytes_transferred)
        return {
            "total_analyses": self.total_analyses,
            "total_transactions": self.total_transactions,
            "total_threads_served": self.total_threads_served,
            "total_bytes_accessed": self.total_bytes_accessed,
            "total_bytes_transferred": self.total_bytes_transferred,
            "total_energy_pj": self.total_energy_pj,
            "avg_transactions_per_warp": avg_transactions,
            "avg_efficiency": avg_efficiency,
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.total_analyses = 0
        self.total_transactions = 0
        self.total_threads_served = 0
        self.total_bytes_accessed = 0
        self.total_bytes_transferred = 0
        self.total_energy_pj = 0.0


def analyze_access_pattern(
    config: SIMTConfig,
    pattern_name: str,
    base_addr: int = 0,
) -> dict[str, Any]:
    """
    Analyze common memory access patterns for educational purposes.

    Args:
        config: SIMT configuration
        pattern_name: One of "coalesced", "strided", "random", "broadcast"
        base_addr: Base address for access

    Returns:
        Analysis results
    """
    import random

    coalescer = MemoryCoalescerSim(config)

    if pattern_name == "coalesced":
        # Best case: consecutive 4-byte accesses
        addresses = [base_addr + tid * 4 for tid in range(32)]
    elif pattern_name == "strided":
        # Worst case: 128-byte stride hits different segments
        addresses = [base_addr + tid * 128 for tid in range(32)]
    elif pattern_name == "random":
        # Random addresses within 1KB range
        random.seed(42)  # Reproducible
        addresses = [base_addr + random.randint(0, 1023) for _ in range(32)]
    elif pattern_name == "broadcast":
        # All threads read same address
        addresses = [base_addr] * 32
    else:
        addresses = [base_addr + tid * 4 for tid in range(32)]

    result = coalescer.analyze(addresses)

    return {
        "pattern": pattern_name,
        "transactions": result.num_transactions,
        "efficiency": result.efficiency,
        "is_coalesced": result.is_fully_coalesced,
        "segments": result.segments,
    }
