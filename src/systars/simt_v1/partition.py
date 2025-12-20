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
from .execution_unit import OPCODE_MAP, ExecutionUnitSim, Opcode
from .load_store_unit import LoadStoreUnitSim
from .operand_collector import OperandCollectorSim
from .register_file import RegisterFileSim
from .shared_memory import SharedMemorySim
from .warp_scheduler import Instruction, WarpSchedulerSim, WarpState


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
    total_memory_stalls: int = 0
    total_lsu_rejections: int = 0  # Count of LSU queue-full rejections
    instruction_counts: dict = field(default_factory=dict)  # Per-opcode counts

    # Pending memory requests (when LSU queue is full)
    # Each entry: (warp_id, instruction, addresses, store_data)
    pending_memory_requests: list = field(default_factory=list)

    # Components (initialized in __post_init__, not passed to __init__)
    scheduler: WarpSchedulerSim = field(init=False)  # type: ignore[assignment]
    register_file: RegisterFileSim = field(init=False)  # type: ignore[assignment]
    operand_collector: OperandCollectorSim = field(init=False)  # type: ignore[assignment]
    execution_unit: ExecutionUnitSim = field(init=False)  # type: ignore[assignment]
    shared_memory: SharedMemorySim = field(init=False)  # type: ignore[assignment]
    load_store_unit: LoadStoreUnitSim = field(init=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize all components."""
        self.scheduler = WarpSchedulerSim(self.config, self.partition_id)
        self.register_file = RegisterFileSim(self.config, self.partition_id)
        self.operand_collector = OperandCollectorSim(self.config, self.partition_id)
        self.execution_unit = ExecutionUnitSim(self.config, self.partition_id)
        self.shared_memory = SharedMemorySim(self.config)
        self.load_store_unit = LoadStoreUnitSim(
            self.config, self.partition_id, max_pending=self.config.lsu_max_pending
        )
        # Connect LSU to shared memory
        self.load_store_unit.shared_memory = self.shared_memory

    def load_program(self, warp_id: int, instructions: list[Instruction]) -> None:
        """Load program into a warp."""
        self.scheduler.load_program(warp_id, instructions)

    def activate_warps(self, num_warps: int) -> None:
        """Activate warps for execution."""
        self.scheduler.activate_warps(num_warps)

    def _is_memory_instruction(self, instruction: Instruction) -> bool:
        """Check if instruction is a memory operation (LD/ST)."""
        opcode = OPCODE_MAP.get(instruction.opcode, Opcode.NOP)
        return opcode in (Opcode.LD, Opcode.ST)

    def _compute_addresses(
        self, _instruction: Instruction, per_thread_operands: list[list[int]]
    ) -> list[int]:
        """
        Compute per-thread addresses for memory instructions.

        Each thread computes its own address from its src1 operand.
        This models true SIMT behavior where threads can have different addresses.

        Args:
            _instruction: Memory instruction
            per_thread_operands: Per-thread operands [thread_id][src_idx]

        Returns:
            List of 32 addresses, one per thread.
        """
        addresses = []
        for thread_id in range(32):
            if thread_id < len(per_thread_operands):
                thread_ops = per_thread_operands[thread_id]
                # src1 (thread_ops[0]) contains the address for this thread
                addr = thread_ops[0] if len(thread_ops) > 0 else 0
            else:
                addr = 0
            addresses.append(addr)
        return addresses

    def step(self) -> dict[str, Any]:
        """
        Execute one cycle.

        Returns:
            Dictionary with cycle status information.
        """
        completed_list: list[tuple[int, int, list[int]]] = []
        conflicts_list: list[int] = []
        memory_completed_list: list[tuple[int, int, list[int]]] = []
        status: dict[str, Any] = {
            "cycle": self.cycle,
            "issued": None,
            "fired": None,
            "completed": completed_list,
            "memory_completed": memory_completed_list,
            "bank_conflicts": conflicts_list,
            "warp_states": self.scheduler.get_warp_states(),
        }

        # Reset register file and shared memory for new cycle
        self.register_file.reset_cycle()
        self.shared_memory.reset_cycle()

        # 0. Process pending register file writes (bank-conflicted)
        # This drains writes that couldn't complete last cycle
        writes_done = self.register_file.process_pending_writes()
        if writes_done > 0:
            # Pending writes consumed bank bandwidth
            pass

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

        # 2. Operand collection: read per-thread values from register file
        conflicts = self.operand_collector.collect_operands(self.register_file)
        if conflicts:
            conflicts_list.extend(conflicts)
            self.total_bank_conflicts += len(conflicts)

        # 3a. Try to issue any pending memory requests first (from LSU queue-full rejections)
        if self.pending_memory_requests:
            pending = self.pending_memory_requests[0]
            warp_id, instruction, addresses, store_data = pending
            if self.load_store_unit.issue(warp_id, instruction, addresses, store_data):
                # Successfully issued pending request
                self.pending_memory_requests.pop(0)
                self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
                self.total_memory_stalls += 1
                self.total_instructions += 1
                opcode_str = instruction.opcode
                self.instruction_counts[opcode_str] = self.instruction_counts.get(opcode_str, 0) + 1
            # If still rejected, leave in queue and try again next cycle

        # 3b. Fire ready instructions to execution unit or LSU
        fire_result = self.operand_collector.fire()
        if fire_result:
            warp_id, instruction, per_thread_operands = fire_result
            status["fired"] = (warp_id, instruction)

            if self._is_memory_instruction(instruction):
                # Route to Load/Store Unit
                addresses = self._compute_addresses(instruction, per_thread_operands)
                opcode = OPCODE_MAP.get(instruction.opcode, Opcode.NOP)

                if opcode == Opcode.ST:
                    # For stores, src2 (index 1) contains per-thread data to write
                    store_data = [ops[1] if len(ops) > 1 else 0 for ops in per_thread_operands]
                else:
                    store_data = None

                # Issue to LSU
                if self.load_store_unit.issue(warp_id, instruction, addresses, store_data):
                    # Mark warp as stalled on memory
                    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
                    self.total_memory_stalls += 1
                    self.total_instructions += 1
                    opcode_str = instruction.opcode
                    self.instruction_counts[opcode_str] = (
                        self.instruction_counts.get(opcode_str, 0) + 1
                    )
                else:
                    # LSU queue full - buffer request and stall warp
                    self.pending_memory_requests.append(
                        (warp_id, instruction, addresses, store_data)
                    )
                    # Stall the warp until the memory request can be issued
                    self.scheduler.warps[warp_id].state = WarpState.STALLED_MEM
                    self.total_lsu_rejections += 1
            else:
                # Route to execution unit with per-thread operands
                self.execution_unit.issue(warp_id, instruction, per_thread_operands)
                self.total_instructions += 1
                opcode_str = instruction.opcode
                self.instruction_counts[opcode_str] = self.instruction_counts.get(opcode_str, 0) + 1

        # 4. Advance execution pipeline
        completed = self.execution_unit.tick()
        for warp_id, dst_reg, per_thread_results in completed:
            # Queue per-thread results for writeback (with bank conflict handling)
            self.register_file.queue_write_all_threads(dst_reg, per_thread_results)
            completed_list.append((warp_id, dst_reg, per_thread_results))

        # 5. Advance LSU and handle memory completions
        mem_completed = self.load_store_unit.tick()
        for warp_id, dst_reg, data in mem_completed:
            # For loads (dst_reg >= 0): queue per-thread data for writeback
            # For stores (dst_reg == -1): no writeback needed
            if dst_reg >= 0 and data:
                # Queue all 32 values for writeback with bank conflict handling
                self.register_file.queue_write_all_threads(dst_reg, data)
            memory_completed_list.append((warp_id, dst_reg, data))
            # Unstall the warp
            warp = self.scheduler.warps[warp_id]
            if warp.state == WarpState.STALLED_MEM:
                if warp.has_more_instructions():
                    warp.state = WarpState.READY
                else:
                    warp.state = WarpState.DONE

        # 6. Update warp scheduler state
        self.scheduler.tick()

        # Check if done (include all pipeline stages, pending writes, and pending memory reqs)
        self.done = (
            self.scheduler.all_done()
            and not self.operand_collector.is_busy()
            and not self.execution_unit.is_busy()
            and not self.load_store_unit.is_busy()
            and not self.register_file.has_pending_writes()
            and not self.pending_memory_requests
        )

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
            + self.shared_memory.total_energy_pj
            + self.load_store_unit.total_energy_pj
            # Add instruction fetch energy
            + self.total_instructions * self.config.instruction_fetch_energy_pj
            + self.total_instructions * self.config.instruction_decode_energy_pj
        )

    def get_statistics(self) -> dict:
        """Get execution statistics."""
        return {
            "cycles": self.cycle,
            "instructions": self.total_instructions,
            "instruction_counts": dict(self.instruction_counts),  # Per-opcode breakdown
            "stalls": self.total_stalls,
            "bank_conflicts": self.total_bank_conflicts,
            "memory_stalls": self.total_memory_stalls,
            "lsu_rejections": self.total_lsu_rejections,  # LSU queue-full rejections
            "energy_pj": self.get_energy_pj(),
            "ipc": self.total_instructions / max(1, self.cycle),
            "lsu_stats": self.load_store_unit.get_statistics(),
            "shared_memory_stats": self.shared_memory.get_statistics(),
        }

    def get_visualization(self) -> dict[str, Any]:
        """Get visualization data for all components."""
        return {
            "warp_states": self.scheduler.get_visualization(),
            "register_banks": self.register_file.get_bank_visualization(),
            "collectors": self.operand_collector.get_visualization(),
            "collector_status": self.operand_collector.get_status(),
            "pipeline": self.execution_unit.get_visualization(),
            "alu_detail": self.execution_unit.get_detailed_visualization(),
            "shared_memory": self.shared_memory.get_visualization(),
            "lsu": self.load_store_unit.get_visualization(),
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


def create_tiled_gemm_program(
    M: int,
    N: int,
    K: int,
    warp_id: int,
    warp_size: int = 32,
) -> list[Instruction]:
    """
    Create a tiled GEMM program for a specific warp.

    In parallel GEMM:
    - Output matrix C is M×N elements
    - Each thread computes one output element C[i,j]
    - Each warp (32 threads) computes 32 output elements
    - This warp computes elements [warp_id*32 : (warp_id+1)*32]

    Thread mapping (row-major):
        thread_id -> (row, col) where:
        global_idx = warp_id * 32 + thread_id
        row = global_idx // N
        col = global_idx % N

    Computes: C[row, col] = sum(A[row, k] * B[k, col]) for k in 0..K

    Register allocation per thread:
        R0: accumulator (partial sum)
        R1: A[row, k] value
        R2: B[k, col] value
        R3: row index (computed from thread_id)
        R4: col index (computed from thread_id)
        R5: temp for address calculation

    Args:
        M, N, K: Matrix dimensions
        warp_id: Which warp this program is for (0, 1, 2, ...)
        warp_size: Threads per warp (default 32)

    Returns:
        List of instructions for this warp's GEMM tile.
    """
    instructions: list[Instruction] = []

    # Calculate which output elements this warp computes
    start_idx = warp_id * warp_size

    # If this warp has no work (output is smaller than warp coverage), return empty
    if start_idx >= M * N:
        return instructions

    # Initialize accumulator to zero
    # Each thread's R0 starts at 0
    instructions.append(Instruction(opcode="MOV", dst=0, src1=0, latency=1))

    # K iterations of the reduction loop
    # Each thread loads its row of A and column of B
    for _k in range(K):
        # Load A[row, k] - each thread loads from its assigned row
        # Address = base_A + row * K + k (thread computes its own address)
        instructions.append(Instruction(opcode="LD", dst=1, src1=3, latency=4))

        # Load B[k, col] - each thread loads from its assigned column
        # Address = base_B + k * N + col
        instructions.append(Instruction(opcode="LD", dst=2, src1=4, latency=4))

        # FMA: R0 = R1 * R2 + R0
        instructions.append(Instruction(opcode="FFMA", dst=0, src1=1, src2=2, src3=0, latency=4))

    # Store result to C[row, col]
    # Address = base_C + row * N + col
    instructions.append(Instruction(opcode="ST", dst=5, src1=0, latency=4))

    return instructions


def get_gemm_warp_count(M: int, N: int, warp_size: int = 32) -> int:
    """
    Calculate number of warps needed for M×N GEMM output.

    Args:
        M, N: Output matrix dimensions
        warp_size: Threads per warp (default 32)

    Returns:
        Number of warps needed.
    """
    total_elements = M * N
    return (total_elements + warp_size - 1) // warp_size


def get_warp_tile_info(
    warp_id: int, M: int, N: int, warp_size: int = 32
) -> tuple[int, int, int, int]:
    """
    Get the tile boundaries for a specific warp.

    Args:
        warp_id: Which warp
        M, N: Output matrix dimensions
        warp_size: Threads per warp

    Returns:
        Tuple of (start_row, start_col, end_row, end_col) for this warp's tile.
    """
    start_idx = warp_id * warp_size
    end_idx = min(start_idx + warp_size, M * N)

    if start_idx >= M * N:
        return (M, N, M, N)  # Empty tile

    start_row = start_idx // N
    start_col = start_idx % N
    end_row = (end_idx - 1) // N
    end_col = (end_idx - 1) % N

    return (start_row, start_col, end_row, end_col)


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
