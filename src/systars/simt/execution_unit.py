"""
Execution Units for SIMT Streaming Multiprocessor.

Each partition has 8 execution units (ALUs), each with a multi-stage pipeline.
This models how a warp of 32 threads executes across 8 ALUs over 4 cycles
(32 threads / 8 ALUs = 4 cycles per warp instruction).

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    EXECUTION UNIT CLUSTER (8 ALUs)                       │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────────┐│
    │  │                      ALU 0 (4-stage pipeline)                       ││
    │  │  [IN]───▶[EX1]───▶[EX2]───▶[EX3]───▶[WB]                           ││
    │  │   W0      W2       --       W1       --                             ││
    │  └─────────────────────────────────────────────────────────────────────┘│
    │  ┌─────────────────────────────────────────────────────────────────────┐│
    │  │                      ALU 1 (4-stage pipeline)                       ││
    │  │  [IN]───▶[EX1]───▶[EX2]───▶[EX3]───▶[WB]                           ││
    │  │   W1      --       W0       --       W3                             ││
    │  └─────────────────────────────────────────────────────────────────────┘│
    │  │  ... (ALUs 2-7) ...                                                 ││
    │  └─────────────────────────────────────────────────────────────────────┘│
    │                                                                          │
    │  Operation Types:                                                        │
    │    INT32 ALU: 1 cycle  (IADD, ISUB, IAND, IOR, IXOR, ISHL, ISHR, MOV)  │
    │    INT32 MUL: 4 cycles (IMUL)                                           │
    │    FP32:      4 cycles (FADD, FSUB, FMUL, FFMA)                         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any

from amaranth import Module, Signal, signed, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig
from .warp_scheduler import Instruction


class Opcode(IntEnum):
    """Simplified opcode encoding."""

    NOP = 0
    # Integer ALU (1 cycle)
    IADD = auto()
    ISUB = auto()
    IAND = auto()
    IOR = auto()
    IXOR = auto()
    ISHL = auto()
    ISHR = auto()
    MOV = auto()
    # Integer MUL (4 cycles)
    IMUL = auto()
    # Float (4 cycles)
    FADD = auto()
    FSUB = auto()
    FMUL = auto()
    FFMA = auto()  # Fused multiply-add: d = a * b + c
    # Comparison
    ISETP = auto()  # Set predicate
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
class PipelineStage:
    """Single stage in an ALU pipeline."""

    valid: bool = False
    warp_id: int = 0
    opcode: Opcode = Opcode.NOP
    dst: int | None = None
    result: int = 0
    cycles_in_stage: int = 0
    total_latency: int = 1


@dataclass
class ALUPipeline:
    """Single ALU with multi-stage pipeline."""

    alu_id: int = 0
    num_stages: int = 4

    # Pipeline stages (index 0 = input, index 3 = output/writeback)
    stages: list[PipelineStage] = field(default_factory=list)

    # Statistics
    cycles_active: int = 0
    operations_completed: int = 0

    def __post_init__(self):
        """Initialize pipeline stages."""
        if not self.stages:
            self.stages = [PipelineStage() for _ in range(self.num_stages)]

    def is_busy(self) -> bool:
        """Check if ALU has any active operations."""
        return any(stage.valid for stage in self.stages)

    def can_accept(self) -> bool:
        """Check if ALU can accept a new instruction."""
        return not self.stages[0].valid

    def issue(
        self,
        warp_id: int,
        opcode: Opcode,
        dst: int | None,
        result: int,
        latency: int,
    ) -> bool:
        """Issue instruction to ALU pipeline."""
        if not self.can_accept():
            return False

        self.stages[0] = PipelineStage(
            valid=True,
            warp_id=warp_id,
            opcode=opcode,
            dst=dst,
            result=result,
            cycles_in_stage=0,
            total_latency=latency,
        )
        return True

    def tick(self) -> tuple[int, int, int] | None:
        """
        Advance pipeline one cycle.

        Returns:
            Completed operation as (warp_id, dst_reg, result) or None.
        """
        completed = None

        # Check writeback stage for completion
        wb_stage = self.stages[-1]
        if wb_stage.valid:
            # Check if enough cycles have passed
            cycles_needed = wb_stage.total_latency
            stages_passed = self.num_stages - 1  # Stages 0 to num_stages-1
            if wb_stage.cycles_in_stage >= (cycles_needed - stages_passed):
                if wb_stage.dst is not None:
                    completed = (wb_stage.warp_id, wb_stage.dst, wb_stage.result)
                wb_stage.valid = False
                self.operations_completed += 1

        # Advance stages from back to front
        for i in range(self.num_stages - 1, 0, -1):
            curr_stage = self.stages[i]
            prev_stage = self.stages[i - 1]

            if not curr_stage.valid and prev_stage.valid:
                # Move instruction forward
                self.stages[i] = PipelineStage(
                    valid=True,
                    warp_id=prev_stage.warp_id,
                    opcode=prev_stage.opcode,
                    dst=prev_stage.dst,
                    result=prev_stage.result,
                    cycles_in_stage=0,
                    total_latency=prev_stage.total_latency,
                )
                prev_stage.valid = False

        # Increment cycle counters for stages that didn't move
        for stage in self.stages:
            if stage.valid:
                stage.cycles_in_stage += 1
                self.cycles_active += 1

        return completed

    def get_visualization(self) -> list[str]:
        """Get visualization of each pipeline stage."""
        result = []
        for stage in self.stages:
            if stage.valid:
                result.append(f"W{stage.warp_id}")
            else:
                result.append("--")
        return result


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
    Behavioral simulation model of execution unit cluster.

    Models 8 ALUs per partition, each with a 4-stage pipeline.
    Used for animation and energy estimation without RTL simulation.
    """

    config: SIMTConfig
    partition_id: int = 0

    # ALU cluster (8 ALUs per partition)
    num_alus: int = 8
    pipeline_depth: int = 4
    alus: list[ALUPipeline] = field(default_factory=list)

    # Round-robin ALU selection
    next_alu: int = 0

    # Statistics
    total_operations: int = 0
    total_alu_ops: int = 0
    total_fma_ops: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self):
        """Initialize ALU cluster."""
        self.num_alus = self.config.cores_per_partition  # 8 cores = 8 ALUs
        self.alus = [
            ALUPipeline(alu_id=i, num_stages=self.pipeline_depth) for i in range(self.num_alus)
        ]

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

        # Compute result
        result = self._compute(
            opcode,
            operands[0] if len(operands) > 0 else 0,
            operands[1] if len(operands) > 1 else 0,
            operands[2] if len(operands) > 2 else 0,
        )

        # Find available ALU (round-robin)
        for _ in range(self.num_alus):
            alu = self.alus[self.next_alu]
            self.next_alu = (self.next_alu + 1) % self.num_alus

            if alu.can_accept():
                alu.issue(warp_id, opcode, instruction.dst, result, latency)

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
                    Opcode.MOV,
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

        return False

    def tick(self) -> list[tuple[int, int, int]]:
        """
        Advance all ALU pipelines one cycle.

        Returns:
            List of completed operations as (warp_id, dst_reg, result).
        """
        completed = []

        for alu in self.alus:
            result = alu.tick()
            if result is not None:
                completed.append(result)

        return completed

    def is_busy(self) -> bool:
        """Check if any ALU has in-flight operations."""
        return any(alu.is_busy() for alu in self.alus)

    def get_utilization(self) -> float:
        """Get ALU utilization as percentage."""
        total_stages = self.num_alus * self.pipeline_depth
        busy_stages = sum(sum(1 for stage in alu.stages if stage.valid) for alu in self.alus)
        return (busy_stages / total_stages) * 100 if total_stages > 0 else 0.0

    def get_visualization(self) -> list[str]:
        """Get simple visualization (first 4 ALUs' first stage for legacy compat)."""
        result = []
        for alu in self.alus[:4]:
            if alu.stages[0].valid:
                stage = alu.stages[0]
                result.append(f"W{stage.warp_id}:{stage.opcode.name}")
            else:
                result.append("----")
        return result

    def get_detailed_visualization(self) -> dict[str, Any]:
        """
        Get detailed visualization of all ALUs and their pipelines.

        Returns:
            Dictionary with per-ALU pipeline state.
        """
        return {
            "num_alus": self.num_alus,
            "pipeline_depth": self.pipeline_depth,
            "utilization": self.get_utilization(),
            "alus": [
                {
                    "alu_id": alu.alu_id,
                    "busy": alu.is_busy(),
                    "stages": alu.get_visualization(),
                    "operations_completed": alu.operations_completed,
                }
                for alu in self.alus
            ],
        }

    def drain(self) -> list[tuple[int, int, int]]:
        """Drain all remaining operations from all ALU pipelines."""
        all_completed = []
        while self.is_busy():
            completed = self.tick()
            all_completed.extend(completed)
        return all_completed
