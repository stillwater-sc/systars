"""
Execution Units for SIMT Streaming Multiprocessor.

Execution units perform the actual computation:
- INT32 ALU: add, sub, mul, shift, logic
- FP32 FMA: fused multiply-add
- SFU: special functions (sin, cos, rsqrt)
- LD/ST: memory operations (not implemented here)

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    EXECUTION UNIT                                │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │                      PIPELINE                              │  │
    │  │                                                            │  │
    │  │   Stage 0      Stage 1      Stage 2      Stage 3          │  │
    │  │  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐         │  │
    │  │  │ IN   │────▶│ EX1  │────▶│ EX2  │────▶│ WB   │         │  │
    │  │  │DECODE│     │ ALU  │     │ FMA  │     │      │         │  │
    │  │  └──────┘     └──────┘     └──────┘     └──────┘         │  │
    │  │                                                            │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                                                                  │
    │  ┌─────────────────────┐  ┌─────────────────────┐              │
    │  │     INT32 ALU       │  │     FP32 FMA        │              │
    │  │  ┌───────────────┐  │  │  ┌───────────────┐  │              │
    │  │  │ ADD/SUB/AND   │  │  │  │ MUL + ADD     │  │              │
    │  │  │ OR/XOR/SHL    │  │  │  │ (pipelined)   │  │              │
    │  │  │ SHR/MUL       │  │  │  │ 4 cycles      │  │              │
    │  │  │ 1 cycle       │  │  │  │               │  │              │
    │  │  └───────────────┘  │  │  └───────────────┘  │              │
    │  └─────────────────────┘  └─────────────────────┘              │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from amaranth import Module, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig
from .warp_scheduler import Instruction


class Opcode(IntEnum):
    """Simplified opcode encoding."""

    NOP = 0
    # Integer ALU
    IADD = auto()
    ISUB = auto()
    IMUL = auto()
    IAND = auto()
    IOR = auto()
    IXOR = auto()
    ISHL = auto()
    ISHR = auto()
    # Float
    FADD = auto()
    FSUB = auto()
    FMUL = auto()
    FFMA = auto()  # Fused multiply-add: d = a * b + c
    # Comparison
    ISETP = auto()  # Set predicate
    # Move
    MOV = auto()
    # Memory (placeholder)
    LD = auto()
    ST = auto()


# Map string opcodes to enum
OPCODE_MAP = {
    "NOP": Opcode.NOP,
    "IADD": Opcode.IADD,
    "ISUB": Opcode.ISUB,
    "IMUL": Opcode.IMUL,
    "IAND": Opcode.IAND,
    "IOR": Opcode.IOR,
    "IXOR": Opcode.IXOR,
    "ISHL": Opcode.ISHL,
    "ISHR": Opcode.ISHR,
    "FADD": Opcode.FADD,
    "FSUB": Opcode.FSUB,
    "FMUL": Opcode.FMUL,
    "FFMA": Opcode.FFMA,
    "ISETP": Opcode.ISETP,
    "MOV": Opcode.MOV,
    "LD": Opcode.LD,
    "ST": Opcode.ST,
}

# Latency per opcode (in cycles)
OPCODE_LATENCY = {
    Opcode.NOP: 1,
    Opcode.IADD: 1,
    Opcode.ISUB: 1,
    Opcode.IMUL: 4,
    Opcode.IAND: 1,
    Opcode.IOR: 1,
    Opcode.IXOR: 1,
    Opcode.ISHL: 1,
    Opcode.ISHR: 1,
    Opcode.FADD: 4,
    Opcode.FSUB: 4,
    Opcode.FMUL: 4,
    Opcode.FFMA: 4,
    Opcode.ISETP: 1,
    Opcode.MOV: 1,
    Opcode.LD: 100,  # Variable, depends on cache
    Opcode.ST: 1,  # Fire and forget
}


@dataclass
class PipelineEntry:
    """Entry in execution pipeline."""

    valid: bool = False
    warp_id: int = 0
    opcode: Opcode = Opcode.NOP
    dst: int | None = None
    src1_data: int = 0
    src2_data: int = 0
    src3_data: int = 0
    result: int = 0
    cycles_remaining: int = 0


class ExecutionUnit(Component):
    """
    RTL execution unit component.

    Pipelined execution with variable latency per operation type.

    Ports:
        # Input from operand collector
        in_valid: Valid instruction to execute
        in_warp: Warp ID
        in_opcode: Operation code
        in_dst: Destination register
        in_src1: Source 1 data
        in_src2: Source 2 data
        in_src3: Source 3 data (for FMA)

        # Output to writeback
        out_valid: Valid result
        out_warp: Warp ID for result
        out_dst: Destination register
        out_data: Result data

        # Status
        busy: Pipeline has in-flight operations
    """

    def __init__(self, config: SIMTConfig, partition_id: int = 0):
        """
        Initialize execution unit.

        Args:
            config: SIMT configuration
            partition_id: Identifier for this partition
        """
        self.config = config
        self.partition_id = partition_id

        self.warp_bits = max(1, (config.max_warps_per_partition - 1).bit_length())
        self.addr_bits = max(1, config.registers_per_partition.bit_length())

        super().__init__(
            {
                # Input
                "in_valid": In(1),
                "in_warp": In(unsigned(self.warp_bits)),
                "in_opcode": In(unsigned(8)),
                "in_dst": In(unsigned(self.addr_bits)),
                "in_src1": In(signed(config.register_width)),
                "in_src2": In(signed(config.register_width)),
                "in_src3": In(signed(config.register_width)),
                # Output
                "out_valid": Out(1),
                "out_warp": Out(unsigned(self.warp_bits)),
                "out_dst": Out(unsigned(self.addr_bits)),
                "out_data": Out(signed(config.register_width)),
                # Status
                "busy": Out(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config

        # Pipeline registers (4 stages)
        pipe_valid = [Signal(name=f"pipe_valid_{i}") for i in range(4)]
        pipe_result = [
            Signal(signed(cfg.register_width), name=f"pipe_result_{i}") for i in range(4)
        ]
        pipe_dst = [Signal(unsigned(self.addr_bits), name=f"pipe_dst_{i}") for i in range(4)]
        pipe_warp = [Signal(unsigned(self.warp_bits), name=f"pipe_warp_{i}") for i in range(4)]

        # Stage 0: Input and compute (combinational ALU result)
        alu_result = Signal(signed(cfg.register_width), name="alu_result")

        with m.Switch(self.in_opcode):
            with m.Case(Opcode.IADD):
                m.d.comb += alu_result.eq(self.in_src1 + self.in_src2)
            with m.Case(Opcode.ISUB):
                m.d.comb += alu_result.eq(self.in_src1 - self.in_src2)
            with m.Case(Opcode.IMUL):
                m.d.comb += alu_result.eq(self.in_src1 * self.in_src2)
            with m.Case(Opcode.IAND):
                m.d.comb += alu_result.eq(self.in_src1 & self.in_src2)
            with m.Case(Opcode.IOR):
                m.d.comb += alu_result.eq(self.in_src1 | self.in_src2)
            with m.Case(Opcode.IXOR):
                m.d.comb += alu_result.eq(self.in_src1 ^ self.in_src2)
            with m.Case(Opcode.ISHL):
                m.d.comb += alu_result.eq(self.in_src1 << self.in_src2[:5])
            with m.Case(Opcode.ISHR):
                m.d.comb += alu_result.eq(self.in_src1 >> self.in_src2[:5])
            with m.Case(Opcode.FADD, Opcode.FSUB, Opcode.FMUL):
                # Simplified: treat as integer for simulation
                m.d.comb += alu_result.eq(self.in_src1 + self.in_src2)
            with m.Case(Opcode.FFMA):
                # FMA: a * b + c
                m.d.comb += alu_result.eq(self.in_src1 * self.in_src2 + self.in_src3)
            with m.Case(Opcode.MOV):
                m.d.comb += alu_result.eq(self.in_src1)
            with m.Default():
                m.d.comb += alu_result.eq(0)

        # Pipeline advance
        m.d.sync += [
            pipe_valid[0].eq(self.in_valid),
            pipe_result[0].eq(alu_result),
            pipe_dst[0].eq(self.in_dst),
            pipe_warp[0].eq(self.in_warp),
        ]

        for i in range(1, 4):
            m.d.sync += [
                pipe_valid[i].eq(pipe_valid[i - 1]),
                pipe_result[i].eq(pipe_result[i - 1]),
                pipe_dst[i].eq(pipe_dst[i - 1]),
                pipe_warp[i].eq(pipe_warp[i - 1]),
            ]

        # Output from last stage
        m.d.comb += [
            self.out_valid.eq(pipe_valid[3]),
            self.out_data.eq(pipe_result[3]),
            self.out_dst.eq(pipe_dst[3]),
            self.out_warp.eq(pipe_warp[3]),
        ]

        # Busy if any stage has valid data
        m.d.comb += self.busy.eq(pipe_valid[0] | pipe_valid[1] | pipe_valid[2] | pipe_valid[3])

        return m


# =============================================================================
# Simulation Model for Animation
# =============================================================================


@dataclass
class ExecutionUnitSim:
    """
    Behavioral simulation model of execution unit.

    Used for animation and energy estimation without RTL simulation.
    """

    config: SIMTConfig
    partition_id: int = 0

    # Pipeline entries
    pipeline: list = field(default_factory=list)
    pipeline_depth: int = 4

    # Statistics
    total_operations: int = 0
    total_alu_ops: int = 0
    total_fma_ops: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self):
        """Initialize pipeline."""
        self.pipeline = [PipelineEntry() for _ in range(self.pipeline_depth)]

    def _compute(
        self,
        opcode: Opcode,
        src1: int,
        src2: int,
        src3: int,
    ) -> int:
        """Compute result for given opcode."""
        if opcode == Opcode.IADD:
            return (src1 + src2) & 0xFFFFFFFF
        elif opcode == Opcode.ISUB:
            return (src1 - src2) & 0xFFFFFFFF
        elif opcode == Opcode.IMUL:
            return (src1 * src2) & 0xFFFFFFFF
        elif opcode == Opcode.IAND:
            return src1 & src2
        elif opcode == Opcode.IOR:
            return src1 | src2
        elif opcode == Opcode.IXOR:
            return src1 ^ src2
        elif opcode == Opcode.ISHL:
            return (src1 << (src2 & 0x1F)) & 0xFFFFFFFF
        elif opcode == Opcode.ISHR:
            return (src1 >> (src2 & 0x1F)) & 0xFFFFFFFF
        elif opcode == Opcode.FADD:
            # Simplified: treat as integer
            return (src1 + src2) & 0xFFFFFFFF
        elif opcode == Opcode.FSUB:
            return (src1 - src2) & 0xFFFFFFFF
        elif opcode == Opcode.FMUL:
            return (src1 * src2) & 0xFFFFFFFF
        elif opcode == Opcode.FFMA:
            return (src1 * src2 + src3) & 0xFFFFFFFF
        elif opcode == Opcode.MOV:
            return src1
        else:
            return 0

    def issue(
        self,
        warp_id: int,
        instruction: Instruction,
        operands: list[int],
    ) -> bool:
        """
        Issue instruction to execution pipeline.

        Args:
            warp_id: Warp issuing the instruction
            instruction: Instruction to execute
            operands: Source operand values [src1, src2, src3]

        Returns:
            True if instruction was accepted.
        """
        opcode = OPCODE_MAP.get(instruction.opcode, Opcode.NOP)
        latency = OPCODE_LATENCY.get(opcode, 1)

        # Compute result immediately (pipelined execution)
        result = self._compute(
            opcode,
            operands[0] if len(operands) > 0 else 0,
            operands[1] if len(operands) > 1 else 0,
            operands[2] if len(operands) > 2 else 0,
        )

        # Insert into pipeline
        entry = PipelineEntry(
            valid=True,
            warp_id=warp_id,
            opcode=opcode,
            dst=instruction.dst,
            src1_data=operands[0] if len(operands) > 0 else 0,
            src2_data=operands[1] if len(operands) > 1 else 0,
            src3_data=operands[2] if len(operands) > 2 else 0,
            result=result,
            cycles_remaining=latency,
        )

        # Insert at front of pipeline
        self.pipeline.insert(0, entry)
        if len(self.pipeline) > self.pipeline_depth:
            self.pipeline.pop()

        # Update statistics
        self.total_operations += 1
        if opcode in (
            Opcode.IADD,
            Opcode.ISUB,
            Opcode.IAND,
            Opcode.IOR,
            Opcode.IXOR,
            Opcode.ISHL,
            Opcode.ISHR,
        ):
            self.total_alu_ops += 1
            self.total_energy_pj += self.config.alu_energy_pj
        elif opcode == Opcode.IMUL:
            self.total_alu_ops += 1
            self.total_energy_pj += self.config.mul_energy_pj
        elif opcode in (Opcode.FADD, Opcode.FSUB, Opcode.FMUL, Opcode.FFMA):
            self.total_fma_ops += 1
            self.total_energy_pj += self.config.fma_energy_pj

        return True

    def tick(self) -> list[tuple[int, int, int]]:
        """
        Advance pipeline one cycle.

        Returns:
            List of completed operations as (warp_id, dst_reg, result).
        """
        completed = []

        # Check for completed operations
        for entry in self.pipeline:
            if entry.valid:
                entry.cycles_remaining -= 1
                if entry.cycles_remaining <= 0:
                    if entry.dst is not None:
                        completed.append((entry.warp_id, entry.dst, entry.result))
                    entry.valid = False

        return completed

    def is_busy(self) -> bool:
        """Check if pipeline has in-flight operations."""
        return any(entry.valid for entry in self.pipeline)

    def get_visualization(self) -> list[str]:
        """Get visualization of pipeline stages."""
        result = []
        for entry in self.pipeline:
            if entry.valid:
                result.append(f"W{entry.warp_id}:{entry.opcode.name}")
            else:
                result.append("----")
        return result

    def drain(self) -> list[tuple[int, int, int]]:
        """Drain all remaining operations from pipeline."""
        all_completed = []
        while self.is_busy():
            completed = self.tick()
            all_completed.extend(completed)
        return all_completed
