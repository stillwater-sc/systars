"""
Barrier Synchronization Unit for SIMT Streaming Multiprocessor.

Implements __syncthreads() for thread block synchronization within an SM.
All threads (warps) in a thread block must reach the barrier before any
can proceed past it.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    BARRIER SYNCHRONIZATION UNIT                      │
    │                                                                      │
    │  ┌──────────────────────────────────────────────────────────────┐   │
    │  │                  BARRIER SLOTS (4-8 barriers)                 │   │
    │  │                                                               │   │
    │  │  Barrier 0: [W0][W1][  ][  ][  ][  ][  ][  ]  (2/8 arrived)  │   │
    │  │  Barrier 1: [W0][W1][W2][W3][W4][W5][W6][W7]  (8/8 RELEASE)  │   │
    │  │  Barrier 2: [  ][  ][  ][  ][  ][  ][  ][  ]  (0/8 arrived)  │   │
    │  │  ...                                                          │   │
    │  └──────────────────────────────────────────────────────────────┘   │
    │                                                                      │
    │  Operations:                                                         │
    │    ARRIVE: Warp marks itself as arrived at barrier                  │
    │    RELEASE: When all expected warps have arrived, release all       │
    │    RESET: Clear barrier for next iteration                          │
    │                                                                      │
    │  __syncthreads() Implementation:                                     │
    │    1. Warp issues BARRIER instruction                               │
    │    2. Scheduler marks warp as STALLED_BARRIER                       │
    │    3. Barrier unit tracks arrival                                    │
    │    4. When all warps arrive, unit signals release                   │
    │    5. Scheduler moves all warps from STALLED_BARRIER to READY       │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from .config import SIMTConfig


class BarrierState(IntEnum):
    """State of a barrier slot."""

    FREE = 0  # Barrier not in use
    WAITING = auto()  # Some warps have arrived, waiting for others
    RELEASING = auto()  # All warps arrived, releasing
    RELEASED = auto()  # Release complete, ready for reset


@dataclass
class BarrierSlot:
    """A single barrier slot tracking warp arrivals."""

    barrier_id: int = 0
    state: BarrierState = BarrierState.FREE

    # Expected number of warps for this barrier
    expected_warps: int = 0

    # Bitmask of arrived warps
    arrived_mask: int = 0

    # Count of arrived warps (derived from mask)
    @property
    def arrived_count(self) -> int:
        """Count number of arrived warps."""
        return bin(self.arrived_mask).count("1")

    @property
    def is_complete(self) -> bool:
        """Check if all expected warps have arrived."""
        return self.arrived_count >= self.expected_warps

    def arrive(self, warp_id: int) -> bool:
        """
        Mark a warp as arrived at this barrier.

        Args:
            warp_id: Warp that has arrived

        Returns:
            True if this was the last warp (barrier complete)
        """
        self.arrived_mask |= 1 << warp_id
        if self.state == BarrierState.FREE:
            self.state = BarrierState.WAITING

        if self.is_complete:
            self.state = BarrierState.RELEASING
            return True
        return False

    def reset(self) -> None:
        """Reset barrier for next use."""
        self.state = BarrierState.FREE
        self.arrived_mask = 0


@dataclass
class BarrierUnitSim:
    """
    Behavioral simulation model for barrier synchronization.

    Manages multiple barrier slots for concurrent barrier operations.
    """

    config: SIMTConfig

    # Maximum concurrent barriers
    num_barriers: int = 8

    # Barrier slots
    barriers: list[BarrierSlot] = field(default_factory=list)

    # Statistics
    total_barriers_executed: int = 0
    total_warp_arrivals: int = 0
    total_stall_cycles: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self) -> None:
        """Initialize barrier slots."""
        self.barriers = [BarrierSlot(barrier_id=i) for i in range(self.num_barriers)]

    def allocate_barrier(self, expected_warps: int) -> int | None:
        """
        Allocate a free barrier slot.

        Args:
            expected_warps: Number of warps expected to arrive

        Returns:
            Barrier ID if allocated, None if no free slots
        """
        for barrier in self.barriers:
            if barrier.state == BarrierState.FREE:
                barrier.expected_warps = expected_warps
                barrier.state = BarrierState.WAITING
                return barrier.barrier_id
        return None

    def arrive(self, barrier_id: int, warp_id: int) -> bool:
        """
        Record warp arrival at a barrier.

        Args:
            barrier_id: Barrier slot ID
            warp_id: Warp that has arrived

        Returns:
            True if this was the last warp and barrier is releasing
        """
        if barrier_id < 0 or barrier_id >= len(self.barriers):
            return False

        barrier = self.barriers[barrier_id]
        self.total_warp_arrivals += 1

        # Energy for barrier check operation
        self.total_energy_pj += self.config.scheduler_energy_pj

        completed = barrier.arrive(warp_id)
        if completed:
            self.total_barriers_executed += 1

        return completed

    def is_releasing(self, barrier_id: int) -> bool:
        """Check if barrier is releasing warps."""
        if barrier_id < 0 or barrier_id >= len(self.barriers):
            return False
        return self.barriers[barrier_id].state == BarrierState.RELEASING

    def get_arrived_warps(self, barrier_id: int) -> list[int]:
        """Get list of warp IDs that have arrived at the barrier."""
        if barrier_id < 0 or barrier_id >= len(self.barriers):
            return []

        mask = self.barriers[barrier_id].arrived_mask
        return [i for i in range(32) if (mask >> i) & 1]

    def release(self, barrier_id: int) -> list[int]:
        """
        Release all warps from a barrier.

        Args:
            barrier_id: Barrier slot ID

        Returns:
            List of warp IDs to release
        """
        if barrier_id < 0 or barrier_id >= len(self.barriers):
            return []

        barrier = self.barriers[barrier_id]
        if barrier.state != BarrierState.RELEASING:
            return []

        # Get warps to release
        warps_to_release = self.get_arrived_warps(barrier_id)

        # Reset barrier for next use
        barrier.reset()

        return warps_to_release

    def tick(self) -> list[tuple[int, list[int]]]:
        """
        Advance barrier unit one cycle.

        Returns:
            List of (barrier_id, [warp_ids]) for barriers that should release
        """
        releases: list[tuple[int, list[int]]] = []

        for barrier in self.barriers:
            if barrier.state == BarrierState.RELEASING:
                warps = self.get_arrived_warps(barrier.barrier_id)
                releases.append((barrier.barrier_id, warps))
                barrier.reset()

        # Track stall cycles for all waiting barriers
        for barrier in self.barriers:
            if barrier.state == BarrierState.WAITING:
                self.total_stall_cycles += barrier.arrived_count

        return releases

    def get_statistics(self) -> dict[str, Any]:
        """Get barrier statistics."""
        return {
            "total_barriers_executed": self.total_barriers_executed,
            "total_warp_arrivals": self.total_warp_arrivals,
            "total_stall_cycles": self.total_stall_cycles,
            "total_energy_pj": self.total_energy_pj,
            "active_barriers": sum(1 for b in self.barriers if b.state != BarrierState.FREE),
        }

    def get_visualization(self) -> list[dict[str, Any]]:
        """Get visualization data for all barriers."""
        return [
            {
                "barrier_id": b.barrier_id,
                "state": b.state.name,
                "expected": b.expected_warps,
                "arrived": b.arrived_count,
                "arrived_mask": f"{b.arrived_mask:08b}",
            }
            for b in self.barriers
            if b.state != BarrierState.FREE
        ]
