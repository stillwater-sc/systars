"""
Streaming Multiprocessor Controller.

Top-level FSM that coordinates all 4 partitions in the SM.
Manages kernel dispatch, thread block allocation, and completion.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    STREAMING MULTIPROCESSOR                      │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │                    SM CONTROLLER                           │  │
    │  │     IDLE → DISPATCH → EXECUTE → BARRIER → COMPLETE        │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
    │  │ PARTITION 0 │  │ PARTITION 1 │  │ PARTITION 2 │  │PARTITION││
    │  │  Warps 0-7  │  │  Warps 8-15 │  │ Warps 16-23 │  │   3     ││
    │  │  8 Cores    │  │  8 Cores    │  │  8 Cores    │  │ Warps   ││
    │  │  16K Regs   │  │  16K Regs   │  │  16K Regs   │  │ 24-31   ││
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘│
    │                                                                  │
    │  Total: 32 Cores, 64K Registers, 32 Concurrent Warps            │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from amaranth import Module, unsigned
from amaranth.lib.wiring import Component, In, Out

from .barrier import BarrierUnitSim
from .config import SIMTConfig
from .global_memory import GlobalMemorySim
from .partition import PartitionSim, create_gemm_program
from .shared_memory import SharedMemorySim
from .warp_scheduler import Instruction


class SMState(IntEnum):
    """SM controller states."""

    IDLE = 0
    DISPATCH = auto()  # Assign thread blocks to partitions
    EXECUTE = auto()  # Run warps
    BARRIER = auto()  # Synchronize at __syncthreads()
    COMPLETE = auto()  # Kernel complete


class StreamingMultiprocessor(Component):
    """
    RTL Streaming Multiprocessor component.

    Top-level SM with 4 partitions.

    Ports:
        # Control
        start: Start kernel execution
        done: Kernel complete

        # Status
        active_warps: Total active warps across all partitions
        total_cycles: Cycles since start
        total_instructions: Instructions executed
    """

    def __init__(self, config: SIMTConfig):
        """
        Initialize SM.

        Args:
            config: SIMT configuration
        """
        self.config = config

        super().__init__(
            {
                # Control
                "start": In(1),
                "done": Out(1),
                # Status
                "active_warps": Out(unsigned(8)),
                "total_cycles": Out(unsigned(32)),
                "total_instructions": Out(unsigned(32)),
            }
        )

    def elaborate(self, _platform):
        m = Module()

        # Partitions would be instantiated here for full RTL
        # For now, this is a placeholder

        return m


# =============================================================================
# Simulation Model for Animation
# =============================================================================


@dataclass
class SMSim:
    """
    Behavioral simulation model of Streaming Multiprocessor.

    Coordinates 4 partitions for cycle-accurate simulation.
    """

    config: SIMTConfig = field(default_factory=SIMTConfig)

    # Partitions
    partitions: list[PartitionSim] = field(default_factory=list)

    # SM-wide shared memory (accessible from all partitions)
    shared_memory: SharedMemorySim = field(init=False)  # type: ignore[assignment]

    # Global memory (DRAM)
    global_memory: GlobalMemorySim = field(init=False)  # type: ignore[assignment]

    # Barrier unit
    barrier_unit: BarrierUnitSim = field(init=False)  # type: ignore[assignment]

    # State
    state: SMState = SMState.IDLE
    cycle: int = 0
    done: bool = False

    # Statistics
    total_instructions: int = 0
    total_stalls: int = 0
    total_bank_conflicts: int = 0

    def __post_init__(self):
        """Initialize all partitions and SM-wide resources."""
        # SM-wide shared memory
        self.shared_memory = SharedMemorySim(self.config)

        # Global memory (DRAM)
        self.global_memory = GlobalMemorySim(self.config)

        # Barrier unit
        self.barrier_unit = BarrierUnitSim(self.config)

        # Create partitions and connect to SM-wide resources
        self.partitions = []
        for i in range(self.config.num_partitions):
            partition = PartitionSim(self.config, partition_id=i)
            # Connect partition's LSU to SM-wide shared memory and global memory
            partition.load_store_unit.shared_memory = self.shared_memory
            partition.load_store_unit.global_memory = self.global_memory
            # Also update partition's shared memory reference
            partition.shared_memory = self.shared_memory
            self.partitions.append(partition)

    def load_program(
        self,
        partition_id: int,
        warp_id: int,
        instructions: list[Instruction],
    ) -> None:
        """Load program into a specific warp."""
        if partition_id < len(self.partitions):
            self.partitions[partition_id].load_program(warp_id, instructions)

    def load_uniform_program(
        self,
        instructions: list[Instruction],
        num_warps: int = 8,
    ) -> None:
        """Load the same program into multiple warps across partitions."""
        warps_per_partition = self.config.max_warps_per_partition

        for i in range(num_warps):
            partition_id = i // warps_per_partition
            local_warp_id = i % warps_per_partition

            if partition_id < len(self.partitions):
                self.partitions[partition_id].load_program(local_warp_id, instructions)

    def activate_warps(self, num_warps: int) -> None:
        """Activate warps for execution."""
        warps_per_partition = self.config.max_warps_per_partition

        for i, partition in enumerate(self.partitions):
            warps_for_this_partition = min(
                warps_per_partition,
                max(0, num_warps - i * warps_per_partition),
            )
            partition.activate_warps(warps_for_this_partition)

        self.state = SMState.EXECUTE

    def step(self) -> dict[str, Any]:
        """
        Execute one cycle across all partitions.

        Returns:
            Dictionary with cycle status information.
        """
        partition_statuses: list[dict[str, Any]] = []
        status: dict[str, Any] = {
            "cycle": self.cycle,
            "state": self.state,
            "partitions": partition_statuses,
        }

        if self.state == SMState.IDLE:
            return status

        if self.state == SMState.EXECUTE:
            # Step all partitions
            all_done = True
            for partition in self.partitions:
                partition_status = partition.step()
                partition_statuses.append(partition_status)
                if not partition.done:
                    all_done = False

            if all_done:
                self.state = SMState.COMPLETE
                self.done = True

        # Update aggregate statistics
        self.total_instructions = sum(p.total_instructions for p in self.partitions)
        self.total_stalls = sum(p.total_stalls for p in self.partitions)
        self.total_bank_conflicts = sum(p.total_bank_conflicts for p in self.partitions)

        self.cycle += 1
        return status

    def run_to_completion(self, max_cycles: int = 10000) -> int:
        """
        Run until all partitions complete.

        Args:
            max_cycles: Maximum cycles before timeout

        Returns:
            Number of cycles executed.
        """
        while not self.done and self.cycle < max_cycles:
            self.step()
        return self.cycle

    def get_energy_pj(self) -> float:
        """Get total energy consumed in pJ."""
        return sum(p.get_energy_pj() for p in self.partitions)

    def get_statistics(self) -> dict:
        """Get execution statistics."""
        return {
            "cycles": self.cycle,
            "instructions": self.total_instructions,
            "stalls": self.total_stalls,
            "bank_conflicts": self.total_bank_conflicts,
            "energy_pj": self.get_energy_pj(),
            "ipc": self.total_instructions / max(1, self.cycle),
            "partition_stats": [p.get_statistics() for p in self.partitions],
        }

    def get_visualization(self) -> dict:
        """Get visualization data for all partitions."""
        return {
            "cycle": self.cycle,
            "state": self.state.name,
            "partitions": [p.get_visualization() for p in self.partitions],
            "energy_pj": self.get_energy_pj(),
            "total_instructions": self.total_instructions,
            "total_stalls": self.total_stalls,
            "total_bank_conflicts": self.total_bank_conflicts,
        }

    def get_active_warp_count(self) -> int:
        """Get total number of active warps."""
        return sum(p.scheduler.get_active_count() for p in self.partitions)


def run_gemm_simulation(
    M: int = 16,
    N: int = 16,
    K: int = 16,
    num_warps: int = 4,
) -> dict:
    """
    Run a GEMM simulation on the SM.

    Args:
        M, N, K: Matrix dimensions
        num_warps: Number of warps to use

    Returns:
        Simulation results.
    """
    config = SIMTConfig()
    sm = SMSim(config)

    # Create GEMM program
    program = create_gemm_program(M, N, K)

    # Load program into warps
    sm.load_uniform_program(program, num_warps)

    # Activate warps
    sm.activate_warps(num_warps)

    # Run to completion
    sm.run_to_completion()

    # Get statistics
    stats = sm.get_statistics()
    stats["matrix_size"] = (M, N, K)
    stats["num_warps"] = num_warps

    return stats
