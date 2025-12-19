"""
Processing Partition for SIMT Streaming Multiprocessor.

A partition (sub-core) is a self-contained processing unit with:
- Warp scheduler (manages 8 warps)
- Register file (16K registers in 16 banks)
- Operand collectors (gather operands before execution)
- Execution units (ALU, FMA)

The SM contains 4 partitions, each processing warps independently.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PARTITION (SUB-CORE)                          │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │                  WARP SCHEDULER                            │  │
    │  │     Warps 0-7   │   Round-Robin   │   Scoreboard          │  │
    │  └─────────────────────────┬─────────────────────────────────┘  │
    │                            │ Issued Instruction                  │
    │                            ▼                                     │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │              REGISTER FILE (16K Registers)                 │  │
    │  │         16 Banks × 1K Registers per Bank                   │  │
    │  └─────────────────────────┬─────────────────────────────────┘  │
    │                            │ Operand Reads                       │
    │                            ▼                                     │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │              OPERAND COLLECTORS (2)                        │  │
    │  │     Gather operands   │   Bank arbitration                 │  │
    │  └─────────────────────────┬─────────────────────────────────┘  │
    │                            │ Ready Instruction + Operands        │
    │                            ▼                                     │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │              EXECUTION UNITS                               │  │
    │  │     INT32 ALU  │  FP32 FMA  │  SFU  │  LD/ST              │  │
    │  └─────────────────────────┬─────────────────────────────────┘  │
    │                            │ Results                             │
    │                            ▼                                     │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │              WRITEBACK                                     │  │
    │  │     Write results to register file                         │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from typing import Any

from amaranth import Module, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig
from .execution_unit import ExecutionUnitSim
from .operand_collector import OperandCollectorSim
from .register_file import RegisterFileSim
from .warp_scheduler import Instruction, WarpSchedulerSim


class Partition(Component):
    """
    RTL partition component.

    Integrates warp scheduler, register file, operand collectors,
    and execution units.

    Ports:
        # Instruction cache interface
        icache_addr: Instruction address
        icache_data: Instruction data
        icache_valid: Instruction valid

        # Control
        enable: Enable partition execution
        start: Start execution
        done: All warps completed

        # Status
        active_warps: Number of active warps
        issued_count: Total instructions issued
        stall_count: Total stall cycles
    """

    def __init__(self, config: SIMTConfig, partition_id: int = 0):
        """
        Initialize partition.

        Args:
            config: SIMT configuration
            partition_id: Identifier for this partition
        """
        self.config = config
        self.partition_id = partition_id

        self.warp_bits = max(1, (config.max_warps_per_partition - 1).bit_length())

        super().__init__(
            {
                # Control
                "enable": In(1),
                "start": In(1),
                "done": Out(1),
                # Status
                "active_warps": Out(unsigned(self.warp_bits + 1)),
                "issued_count": Out(unsigned(32)),
                "stall_count": Out(unsigned(32)),
            }
        )

    def elaborate(self, _platform):
        m = Module()

        # Submodules would be instantiated here for full RTL
        # For now, this is a placeholder

        return m


# =============================================================================
# Simulation Model for Animation
# =============================================================================


@dataclass
class PartitionSim:
    """
    Behavioral simulation model of a partition.

    Integrates all partition components for cycle-accurate simulation.
    """

    config: SIMTConfig
    partition_id: int = 0

    # State
    cycle: int = 0
    done: bool = False

    # Statistics
    total_instructions: int = 0
    total_stalls: int = 0
    total_bank_conflicts: int = 0

    # Components (initialized in __post_init__, not passed to __init__)
    scheduler: WarpSchedulerSim = field(init=False)  # type: ignore[assignment]
    register_file: RegisterFileSim = field(init=False)  # type: ignore[assignment]
    operand_collector: OperandCollectorSim = field(init=False)  # type: ignore[assignment]
    execution_unit: ExecutionUnitSim = field(init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize all components."""
        self.scheduler = WarpSchedulerSim(self.config, self.partition_id)
        self.register_file = RegisterFileSim(self.config, self.partition_id)
        self.operand_collector = OperandCollectorSim(self.config, self.partition_id)
        self.execution_unit = ExecutionUnitSim(self.config, self.partition_id)

    def load_program(self, warp_id: int, instructions: list[Instruction]) -> None:
        """Load program into a warp."""
        self.scheduler.load_program(warp_id, instructions)

    def activate_warps(self, num_warps: int) -> None:
        """Activate warps for execution."""
        self.scheduler.activate_warps(num_warps)

    def step(self) -> dict[str, Any]:
        """
        Execute one cycle.

        Returns:
            Dictionary with cycle status information.
        """
        completed_list: list[tuple[int, int, int]] = []
        conflicts_list: list[int] = []
        status: dict[str, Any] = {
            "cycle": self.cycle,
            "issued": None,
            "fired": None,
            "completed": completed_list,
            "bank_conflicts": conflicts_list,
            "warp_states": self.scheduler.get_warp_states(),
        }

        # Reset register file for new cycle
        self.register_file.reset_cycle()

        # 1. Warp scheduler: select and issue instruction
        issue_result = self.scheduler.schedule()
        if issue_result:
            warp_id, instruction = issue_result
            status["issued"] = (warp_id, instruction)

            # Allocate operand collector
            collector_id = self.operand_collector.allocate(warp_id, instruction)
            if collector_id is None:
                self.total_stalls += 1
        else:
            self.total_stalls += 1

        # 2. Operand collection: read from register file
        conflicts = self.operand_collector.collect_operands(self.register_file.read)
        if conflicts:
            conflicts_list.extend(conflicts)
            self.total_bank_conflicts += len(conflicts)

        # 3. Fire ready instructions to execution unit
        fire_result = self.operand_collector.fire()
        if fire_result:
            warp_id, instruction, operands = fire_result
            status["fired"] = (warp_id, instruction)
            self.execution_unit.issue(warp_id, instruction, operands)
            self.total_instructions += 1

        # 4. Advance execution pipeline
        completed = self.execution_unit.tick()
        for warp_id, dst_reg, result in completed:
            self.register_file.write(dst_reg, result)
            completed_list.append((warp_id, dst_reg, result))

        # 5. Update warp scheduler state
        self.scheduler.tick()

        # Check if done
        self.done = self.scheduler.all_done() and not self.execution_unit.is_busy()

        self.cycle += 1
        return status

    def run_to_completion(self, max_cycles: int = 10000) -> int:
        """
        Run until all warps complete.

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
        return (
            self.scheduler.total_energy_pj
            + self.register_file.total_energy_pj
            + self.operand_collector.total_energy_pj
            + self.execution_unit.total_energy_pj
            # Add instruction fetch energy
            + self.total_instructions * self.config.instruction_fetch_energy_pj
            + self.total_instructions * self.config.instruction_decode_energy_pj
        )

    def get_statistics(self) -> dict:
        """Get execution statistics."""
        return {
            "cycles": self.cycle,
            "instructions": self.total_instructions,
            "stalls": self.total_stalls,
            "bank_conflicts": self.total_bank_conflicts,
            "energy_pj": self.get_energy_pj(),
            "ipc": self.total_instructions / max(1, self.cycle),
        }

    def get_visualization(self) -> dict:
        """Get visualization data for all components."""
        return {
            "warp_states": self.scheduler.get_visualization(),
            "register_banks": self.register_file.get_bank_visualization(),
            "collectors": self.operand_collector.get_visualization(),
            "collector_status": self.operand_collector.get_status(),
            "pipeline": self.execution_unit.get_visualization(),
        }


def create_gemm_program(
    _M: int,
    _N: int,
    K: int,
    _registers_per_thread: int = 32,
) -> list[Instruction]:
    """
    Create a simple GEMM program for a single thread.

    Computes C[i,j] = sum(A[i,k] * B[k,j]) for all k.

    Args:
        _M, _N, K: Matrix dimensions (M and N are placeholders for API compatibility)
        _registers_per_thread: Available registers (placeholder for API compatibility)

    Returns:
        List of instructions for GEMM.
    """
    instructions = []

    # Simplified GEMM: accumulate K products
    # R0: accumulator
    # R1: A element
    # R2: B element

    # Initialize accumulator
    instructions.append(Instruction(opcode="MOV", dst=0, src1=0, latency=1))

    # K iterations of multiply-accumulate
    for _k in range(K):
        # Load A (simulated with MOV)
        instructions.append(Instruction(opcode="MOV", dst=1, src1=1, latency=1))
        # Load B (simulated with MOV)
        instructions.append(Instruction(opcode="MOV", dst=2, src1=2, latency=1))
        # FMA: R0 = R1 * R2 + R0
        instructions.append(Instruction(opcode="FFMA", dst=0, src1=1, src2=2, src3=0, latency=4))

    return instructions


def create_test_program(num_instructions: int = 10) -> list[Instruction]:
    """
    Create a simple test program with mixed operations.

    Args:
        num_instructions: Number of instructions to generate

    Returns:
        List of test instructions.
    """
    instructions = []

    for i in range(num_instructions):
        # Alternate between different operation types
        if i % 4 == 0:
            # Integer add
            instructions.append(
                Instruction(
                    opcode="IADD",
                    dst=i % 16,
                    src1=(i + 1) % 16,
                    src2=(i + 2) % 16,
                    latency=1,
                )
            )
        elif i % 4 == 1:
            # Integer multiply
            instructions.append(
                Instruction(
                    opcode="IMUL",
                    dst=i % 16,
                    src1=(i + 1) % 16,
                    src2=(i + 2) % 16,
                    latency=4,
                )
            )
        elif i % 4 == 2:
            # Float add
            instructions.append(
                Instruction(
                    opcode="FADD",
                    dst=i % 16,
                    src1=(i + 1) % 16,
                    src2=(i + 2) % 16,
                    latency=4,
                )
            )
        else:
            # FMA
            instructions.append(
                Instruction(
                    opcode="FFMA",
                    dst=i % 16,
                    src1=(i + 1) % 16,
                    src2=(i + 2) % 16,
                    src3=(i + 3) % 16,
                    latency=4,
                )
            )

    return instructions
