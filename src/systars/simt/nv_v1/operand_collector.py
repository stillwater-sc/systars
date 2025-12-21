"""
Operand Collector for SIMT Streaming Multiprocessor.

The operand collector buffers source operands from the register file
until all operands for an instruction are ready, then fires the
instruction to the execution unit.

This component is critical for understanding SIMT overhead:
- Each instruction requires 2-3 register file reads
- Bank conflicts cause multi-cycle stalls
- Collectors provide latency hiding with multiple in-flight instructions

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    OPERAND COLLECTOR UNIT                        │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │                 COLLECTOR 0                                │  │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                      │  │
    │  │  │  SRC1   │ │  SRC2   │ │  SRC3   │  Instruction         │  │
    │  │  │ [READY] │ │ [WAIT]  │ │ [READY] │  FADD R4, R1, R2     │  │
    │  │  │ val=42  │ │ val=?   │ │ val=10  │                      │  │
    │  │  └─────────┘ └─────────┘ └─────────┘                      │  │
    │  │                                                            │  │
    │  │  Status: COLLECTING (2/3 operands ready)                   │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │                 COLLECTOR 1                                │  │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                      │  │
    │  │  │  SRC1   │ │  SRC2   │ │  SRC3   │  Instruction         │  │
    │  │  │ [READY] │ │ [READY] │ │ [READY] │  FMUL R8, R5, R6     │  │
    │  │  │ val=3   │ │ val=7   │ │ val=0   │                      │  │
    │  │  └─────────┘ └─────────┘ └─────────┘                      │  │
    │  │                                                            │  │
    │  │  Status: READY TO FIRE!                                    │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │               REGISTER FILE ARBITER                        │  │
    │  │  Pending requests: B2←C0.src2, B5←C1.src1                 │  │
    │  │  Bank conflicts: B2 (C0, C1 both need)                    │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from amaranth import Module, Signal, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig
from .warp_scheduler import Instruction


class CollectorState(IntEnum):
    """State of an operand collector slot."""

    EMPTY = 0  # No instruction allocated
    COLLECTING = auto()  # Gathering operands
    READY = auto()  # All operands ready, waiting to fire
    FIRED = auto()  # Fired to execution unit


class OperandState(IntEnum):
    """State of a single operand slot."""

    EMPTY = 0  # Not needed
    PENDING = auto()  # Waiting for register file read request
    READING = auto()  # Register file read in progress (1-cycle latency)
    READY = auto()  # Data received


@dataclass
class OperandSlot:
    """Single operand slot in a collector."""

    state: OperandState = OperandState.EMPTY
    register_addr: int | None = None
    value: int = 0
    bank_id: int = 0
    pending_value: int = 0  # Value waiting to be latched after read latency

    def needs_fetch(self) -> bool:
        """Check if this slot needs to fetch from register file."""
        return self.state == OperandState.PENDING

    def is_reading(self) -> bool:
        """Check if this slot is waiting for read to complete."""
        return self.state == OperandState.READING


@dataclass
class CollectorEntry:
    """Entry in an operand collector.

    In SIMT, each warp instruction needs operands for all 32 threads.
    Each thread needs up to 3 source operands (src1, src2, src3).
    Total: 32 threads × 3 sources = 96 operand slots per collector.

    The operand slots are organized as thread_operands[thread_id][src_idx].
    """

    collector_id: int
    warp_size: int = 32  # Number of threads per warp
    state: CollectorState = CollectorState.EMPTY
    warp_id: int = 0
    instruction: Instruction | None = None
    # Per-thread operands: thread_operands[thread_id][src_idx]
    thread_operands: list = field(default_factory=list)
    destination: int | None = None
    cycles_waiting: int = 0

    def __post_init__(self):
        """Initialize operand slots for all threads."""
        if not self.thread_operands:
            # 32 threads × 3 sources = 96 operand slots
            self.thread_operands = [
                [OperandSlot() for _ in range(3)] for _ in range(self.warp_size)
            ]

    def is_ready(self) -> bool:
        """Check if all needed operands for all threads are ready."""
        for thread_ops in self.thread_operands:
            for op in thread_ops:
                if op.state not in (OperandState.EMPTY, OperandState.READY):
                    return False
        return True

    def pending_banks(self) -> list[int]:
        """Get list of banks we're waiting to read from (across all threads)."""
        banks = []
        for thread_ops in self.thread_operands:
            for op in thread_ops:
                if op.state == OperandState.PENDING:
                    banks.append(op.bank_id)
        return banks

    def reading_count(self) -> int:
        """Get count of operands waiting for read to complete."""
        count = 0
        for thread_ops in self.thread_operands:
            for op in thread_ops:
                if op.state == OperandState.READING:
                    count += 1
        return count

    def get_source_progress(self, src_idx: int) -> tuple[int, int, int]:
        """Get progress for a specific source operand across all threads.

        Returns:
            Tuple of (ready_count, reading_count, pending_count)
        """
        ready = reading = pending = 0
        for thread_ops in self.thread_operands:
            op = thread_ops[src_idx]
            if op.state == OperandState.READY:
                ready += 1
            elif op.state == OperandState.READING:
                reading += 1
            elif op.state == OperandState.PENDING:
                pending += 1
        return ready, reading, pending

    def get_visualization(self) -> str:
        """Get compact visualization of operand collection progress.

        Shows per-source progress as a mini progress bar.
        Each source shows: ready/total or a state indicator.
        """
        result = []
        for src_idx in range(3):
            ready, reading, pending = self.get_source_progress(src_idx)
            total_needed = ready + reading + pending
            if total_needed == 0:
                result.append("·")  # Source not needed
            elif ready == self.warp_size:
                result.append("■")  # All threads ready
            elif reading > 0:
                result.append("◐")  # Reading in progress
            elif pending > 0:
                result.append("□")  # Pending
            else:
                result.append("■")  # Ready
        return "".join(result)


class OperandCollector(Component):
    """
    RTL operand collector component.

    Manages multiple collector entries for latency hiding.
    Arbitrates register file access among collectors.

    Ports:
        # Instruction input (from scheduler)
        instr_valid: Valid instruction to collect operands for
        instr_warp: Warp ID for instruction
        instr_dst: Destination register
        instr_src1: Source register 1
        instr_src2: Source register 2
        instr_src3: Source register 3
        instr_opcode: Operation code (encoded)

        # Register file interface
        rf_read0_addr: Address for read port 0
        rf_read0_en: Enable read port 0
        rf_read0_data: Data from read port 0
        rf_read0_valid: Read port 0 valid

        rf_read1_addr: Address for read port 1
        rf_read1_en: Enable read port 1
        rf_read1_data: Data from read port 1
        rf_read1_valid: Read port 1 valid

        rf_read2_addr: Address for read port 2
        rf_read2_en: Enable read port 2
        rf_read2_data: Data from read port 2
        rf_read2_valid: Read port 2 valid

        # Execution unit output
        exec_valid: Valid instruction ready to execute
        exec_warp: Warp ID for execution
        exec_opcode: Operation code
        exec_src1_data: Source 1 data
        exec_src2_data: Source 2 data
        exec_src3_data: Source 3 data
        exec_dst: Destination register

        # Status
        stall: Cannot accept new instruction (all collectors busy)
        bank_conflict: Bank conflict detected
    """

    def __init__(self, config: SIMTConfig, partition_id: int = 0):
        """
        Initialize operand collector.

        Args:
            config: SIMT configuration
            partition_id: Identifier for this partition
        """
        self.config = config
        self.partition_id = partition_id

        self.addr_bits = max(1, config.registers_per_partition.bit_length())
        self.warp_bits = max(1, (config.max_warps_per_partition - 1).bit_length())

        super().__init__(
            {
                # Instruction input
                "instr_valid": In(1),
                "instr_warp": In(unsigned(self.warp_bits)),
                "instr_dst": In(unsigned(self.addr_bits)),
                "instr_src1": In(unsigned(self.addr_bits)),
                "instr_src2": In(unsigned(self.addr_bits)),
                "instr_src3": In(unsigned(self.addr_bits)),
                "instr_src1_valid": In(1),
                "instr_src2_valid": In(1),
                "instr_src3_valid": In(1),
                "instr_opcode": In(unsigned(8)),
                # Register file read ports
                "rf_read0_addr": Out(unsigned(self.addr_bits)),
                "rf_read0_en": Out(1),
                "rf_read0_data": In(unsigned(config.register_width)),
                "rf_read0_valid": In(1),
                "rf_read1_addr": Out(unsigned(self.addr_bits)),
                "rf_read1_en": Out(1),
                "rf_read1_data": In(unsigned(config.register_width)),
                "rf_read1_valid": In(1),
                "rf_read2_addr": Out(unsigned(self.addr_bits)),
                "rf_read2_en": Out(1),
                "rf_read2_data": In(unsigned(config.register_width)),
                "rf_read2_valid": In(1),
                # Execution unit output
                "exec_valid": Out(1),
                "exec_warp": Out(unsigned(self.warp_bits)),
                "exec_opcode": Out(unsigned(8)),
                "exec_src1_data": Out(unsigned(config.register_width)),
                "exec_src2_data": Out(unsigned(config.register_width)),
                "exec_src3_data": Out(unsigned(config.register_width)),
                "exec_dst": Out(unsigned(self.addr_bits)),
                # Status
                "stall": Out(1),
                "has_ready": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        num_collectors = cfg.collectors_per_partition

        # Collector state machines (simplified for RTL)
        collector_valid = [Signal(name=f"collector_{i}_valid") for i in range(num_collectors)]
        collector_ready = [Signal(name=f"collector_{i}_ready") for i in range(num_collectors)]

        # Find empty collector for new instruction
        empty_collector = Signal(unsigned(max(1, (num_collectors - 1).bit_length())))
        found_empty = Signal(name="found_empty")

        for i in range(num_collectors):
            with m.If(~collector_valid[i] & ~found_empty):
                m.d.comb += [
                    empty_collector.eq(i),
                    found_empty.eq(1),
                ]

        # Stall if no empty collector
        m.d.comb += self.stall.eq(self.instr_valid & ~found_empty)

        # Find ready collector to fire
        ready_collector = Signal(unsigned(max(1, (num_collectors - 1).bit_length())))
        found_ready = Signal(name="found_ready")

        for i in range(num_collectors):
            with m.If(collector_ready[i] & ~found_ready):
                m.d.comb += [
                    ready_collector.eq(i),
                    found_ready.eq(1),
                ]

        m.d.comb += self.has_ready.eq(found_ready)
        m.d.comb += self.exec_valid.eq(found_ready)

        return m


# =============================================================================
# Simulation Model for Animation
# =============================================================================


@dataclass
class OperandCollectorSim:
    """
    Behavioral simulation model of operand collector.

    Used for animation and energy estimation without RTL simulation.
    """

    config: SIMTConfig
    partition_id: int = 0

    # Collector entries
    collectors: list = field(default_factory=list)

    # Statistics
    total_fires: int = 0
    total_stalls: int = 0
    total_bank_conflicts: int = 0
    total_energy_pj: float = 0.0
    cycles_collecting: int = 0

    def __post_init__(self):
        """Initialize collector entries."""
        warp_size = self.config.warp_size
        self.collectors = [
            CollectorEntry(collector_id=i, warp_size=warp_size)
            for i in range(self.config.collectors_per_partition)
        ]

    def _get_bank(self, reg_addr: int, thread_id: int) -> int:
        """Get bank ID for a register address.

        In SIMT, each thread's register is in a different bank to enable
        conflict-free access when all threads access the same register number.
        bank_id = (reg_addr + thread_id) % num_banks
        """
        return (reg_addr + thread_id) % self.config.register_banks_per_partition

    def allocate(
        self,
        warp_id: int,
        instruction: Instruction,
    ) -> int | None:
        """
        Allocate a collector for a new instruction.

        In SIMT, this allocates operand collection for all 32 threads.
        Each thread needs its own operand values from the register file.

        Args:
            warp_id: Warp issuing the instruction
            instruction: Instruction to collect operands for

        Returns:
            Collector ID if allocated, None if all busy.
        """
        # Find empty collector
        for collector in self.collectors:
            if collector.state == CollectorState.EMPTY:
                collector.state = CollectorState.COLLECTING
                collector.warp_id = warp_id
                collector.instruction = instruction
                collector.destination = instruction.dst
                collector.cycles_waiting = 0

                # Initialize operand slots for all 32 threads
                sources = [instruction.src1, instruction.src2, instruction.src3]
                for thread_id in range(collector.warp_size):
                    for src_idx, src in enumerate(sources):
                        op = collector.thread_operands[thread_id][src_idx]
                        if src is not None:
                            op.state = OperandState.PENDING
                            op.register_addr = src
                            op.bank_id = self._get_bank(src, thread_id)
                        else:
                            op.state = OperandState.EMPTY
                            op.register_addr = None

                return collector.collector_id

        self.total_stalls += 1
        return None

    def collect_operands(
        self,
        register_file,
    ) -> list[int]:
        """
        Attempt to collect pending operands from register file for all threads.

        This implements proper 2-phase collection with 1-cycle latency:
        - Phase 1: PENDING -> READING (request read from register file)
        - Phase 2: READING -> READY (latch the read data)

        In SIMT, all 32 threads' operands are collected in parallel.
        Bank conflicts occur when multiple threads access the same bank.

        Args:
            register_file: RegisterFileSim with per-thread read capability

        Returns:
            List of banks that had conflicts.
        """
        all_conflicts = []

        for collector in self.collectors:
            if collector.state != CollectorState.COLLECTING:
                continue

            # Phase 2: Latch data for all threads' operands that were READING
            for thread_ops in collector.thread_operands:
                for op in thread_ops:
                    if op.state == OperandState.READING:
                        op.value = op.pending_value
                        op.state = OperandState.READY
                        self.total_energy_pj += self.config.operand_collect_energy_pj

            # Phase 1: Collect pending reads across all threads
            # Group by bank to detect conflicts
            pending_by_bank: dict[int, list[tuple[int, int, OperandSlot]]] = {}
            for thread_id, thread_ops in enumerate(collector.thread_operands):
                for src_idx, op in enumerate(thread_ops):
                    if op.state == OperandState.PENDING:
                        bank = op.bank_id
                        if bank not in pending_by_bank:
                            pending_by_bank[bank] = []
                        pending_by_bank[bank].append((thread_id, src_idx, op))

            # Process reads - first access to each bank succeeds, others conflict
            conflict_banks = []
            for bank, ops_list in pending_by_bank.items():
                if len(ops_list) > 1:
                    # Bank conflict - only first access succeeds this cycle
                    conflict_banks.append(bank)
                    self.total_bank_conflicts += len(ops_list) - 1

                # First access always succeeds
                thread_id, src_idx, op = ops_list[0]
                # Read per-thread value from register file
                if op.register_addr is not None:
                    op.pending_value = register_file.read_thread(op.register_addr, thread_id)
                else:
                    op.pending_value = 0
                op.state = OperandState.READING

            if conflict_banks:
                all_conflicts.extend(conflict_banks)
                self.total_energy_pj += len(conflict_banks) * self.config.bank_conflict_energy_pj

            collector.cycles_waiting += 1

            # Check if all operands for all threads are ready
            if collector.is_ready():
                collector.state = CollectorState.READY

        self.cycles_collecting += 1
        return all_conflicts

    def fire(self) -> tuple[int, Instruction, list[list[int]]] | None:
        """
        Fire a ready instruction to execution unit.

        In SIMT, all 32 threads fire together. Returns per-thread operand values.

        Returns:
            Tuple of (warp_id, instruction, per_thread_operands) where
            per_thread_operands[thread_id][src_idx] is the operand value.
            Returns None if no collector is ready.
        """
        for collector in self.collectors:
            if collector.state == CollectorState.READY:
                # Get operand values for all 32 threads
                # per_thread_operands[thread_id][src_idx]
                per_thread_operands = []
                for thread_id in range(collector.warp_size):
                    thread_ops = collector.thread_operands[thread_id]
                    thread_values = [
                        op.value if op.state == OperandState.READY else 0 for op in thread_ops
                    ]
                    per_thread_operands.append(thread_values)

                result = (collector.warp_id, collector.instruction, per_thread_operands)

                # Reset collector for all threads
                collector.state = CollectorState.EMPTY
                collector.instruction = None
                collector.warp_id = 0
                for thread_ops in collector.thread_operands:
                    for op in thread_ops:
                        op.state = OperandState.EMPTY
                        op.register_addr = None
                        op.value = 0

                self.total_fires += 1
                return result

        return None

    def has_ready(self) -> bool:
        """Check if any collector is ready to fire."""
        return any(c.state == CollectorState.READY for c in self.collectors)

    def has_empty(self) -> bool:
        """Check if any collector is empty."""
        return any(c.state == CollectorState.EMPTY for c in self.collectors)

    def is_busy(self) -> bool:
        """Check if any collector is actively collecting or ready to fire."""
        return any(
            c.state in (CollectorState.COLLECTING, CollectorState.READY) for c in self.collectors
        )

    def get_visualization(self) -> list[str]:
        """Get visualization of all collectors."""
        return [c.get_visualization() for c in self.collectors]

    def get_status(self) -> list[str]:
        """Get status strings for all collectors.

        Shows per-source progress across all 32 threads.
        """
        status = []
        for c in self.collectors:
            if c.state == CollectorState.EMPTY:
                status.append("empty")
            elif c.state == CollectorState.COLLECTING:
                # Count total ready/reading/pending across all threads and sources
                ready = reading = pending = 0
                for thread_ops in c.thread_operands:
                    for op in thread_ops:
                        if op.state == OperandState.READY:
                            ready += 1
                        elif op.state == OperandState.READING:
                            reading += 1
                        elif op.state == OperandState.PENDING:
                            pending += 1
                total = ready + reading + pending
                if reading > 0:
                    status.append(f"{ready}+{reading}/{total}")
                else:
                    status.append(f"{ready}/{total}")
            elif c.state == CollectorState.READY:
                status.append("FIRE!")
            else:
                status.append("???")
        return status
