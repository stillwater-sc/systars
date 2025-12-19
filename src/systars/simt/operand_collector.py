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
    PENDING = auto()  # Waiting for register file
    READY = auto()  # Data received


@dataclass
class OperandSlot:
    """Single operand slot in a collector."""

    state: OperandState = OperandState.EMPTY
    register_addr: int | None = None
    value: int = 0
    bank_id: int = 0

    def needs_fetch(self) -> bool:
        """Check if this slot needs to fetch from register file."""
        return self.state == OperandState.PENDING


@dataclass
class CollectorEntry:
    """Entry in an operand collector."""

    collector_id: int
    state: CollectorState = CollectorState.EMPTY
    warp_id: int = 0
    instruction: Instruction | None = None
    operands: list = field(default_factory=list)  # List of OperandSlot
    destination: int | None = None
    cycles_waiting: int = 0

    def __post_init__(self):
        """Initialize operand slots."""
        if not self.operands:
            self.operands = [OperandSlot() for _ in range(3)]

    def is_ready(self) -> bool:
        """Check if all needed operands are ready."""
        return all(op.state != OperandState.PENDING for op in self.operands)

    def pending_banks(self) -> list[int]:
        """Get list of banks we're waiting to read from."""
        return [op.bank_id for op in self.operands if op.state == OperandState.PENDING]

    def get_visualization(self) -> str:
        """Get visualization of operand collection progress."""
        symbols = []
        for op in self.operands:
            if op.state == OperandState.EMPTY:
                symbols.append("·")
            elif op.state == OperandState.PENDING:
                symbols.append("□")
            else:  # READY
                symbols.append("■")
        return "".join(symbols)


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
        self.collectors = [
            CollectorEntry(collector_id=i) for i in range(self.config.collectors_per_partition)
        ]

    def _get_bank(self, reg_addr: int) -> int:
        """Get bank ID for a register address."""
        return reg_addr % self.config.register_banks_per_partition

    def allocate(
        self,
        warp_id: int,
        instruction: Instruction,
    ) -> int | None:
        """
        Allocate a collector for a new instruction.

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

                # Initialize operand slots
                for i, src in enumerate([instruction.src1, instruction.src2, instruction.src3]):
                    if src is not None:
                        collector.operands[i].state = OperandState.PENDING
                        collector.operands[i].register_addr = src
                        collector.operands[i].bank_id = self._get_bank(src)
                    else:
                        collector.operands[i].state = OperandState.EMPTY
                        collector.operands[i].register_addr = None

                return collector.collector_id

        self.total_stalls += 1
        return None

    def collect_operands(
        self,
        register_read_func,
    ) -> list[int]:
        """
        Attempt to collect pending operands from register file.

        Args:
            register_read_func: Function to read registers, returns (data, conflict, conflict_banks)

        Returns:
            List of banks that had conflicts.
        """
        all_conflicts = []

        for collector in self.collectors:
            if collector.state != CollectorState.COLLECTING:
                continue

            # Gather pending register addresses
            pending_addrs = [
                op.register_addr for op in collector.operands if op.state == OperandState.PENDING
            ]

            if not pending_addrs:
                # All operands ready
                collector.state = CollectorState.READY
                continue

            # Read from register file
            data, has_conflict, conflict_banks = register_read_func(pending_addrs)

            if has_conflict:
                self.total_bank_conflicts += len(conflict_banks)
                all_conflicts.extend(conflict_banks)
                self.total_energy_pj += len(conflict_banks) * self.config.bank_conflict_energy_pj

            # Update operand slots with received data
            data_idx = 0
            for op in collector.operands:
                if op.state == OperandState.PENDING:
                    # Check if this operand's bank had a conflict
                    if (
                        op.bank_id not in conflict_banks
                        or pending_addrs.index(op.register_addr) == 0
                    ):
                        # Received data (first accessor wins on conflict)
                        op.value = data[data_idx]
                        op.state = OperandState.READY
                        self.total_energy_pj += self.config.operand_collect_energy_pj
                    data_idx += 1

            collector.cycles_waiting += 1

            # Check if all operands now ready
            if collector.is_ready():
                collector.state = CollectorState.READY

        self.cycles_collecting += 1
        return all_conflicts

    def fire(self) -> tuple[int, Instruction, list[int]] | None:
        """
        Fire a ready instruction to execution unit.

        Returns:
            Tuple of (warp_id, instruction, operand_values) or None.
        """
        for collector in self.collectors:
            if collector.state == CollectorState.READY:
                # Get operand values
                operand_values = [
                    op.value if op.state == OperandState.READY else 0 for op in collector.operands
                ]

                result = (collector.warp_id, collector.instruction, operand_values)

                # Reset collector
                collector.state = CollectorState.EMPTY
                collector.instruction = None
                collector.warp_id = 0
                for op in collector.operands:
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

    def get_visualization(self) -> list[str]:
        """Get visualization of all collectors."""
        return [c.get_visualization() for c in self.collectors]

    def get_status(self) -> list[str]:
        """Get status strings for all collectors."""
        status = []
        for c in self.collectors:
            if c.state == CollectorState.EMPTY:
                status.append("empty")
            elif c.state == CollectorState.COLLECTING:
                ready = sum(1 for op in c.operands if op.state == OperandState.READY)
                total = sum(1 for op in c.operands if op.state != OperandState.EMPTY)
                status.append(f"{ready}/{total}")
            elif c.state == CollectorState.READY:
                status.append("FIRE!")
            else:
                status.append("???")
        return status
