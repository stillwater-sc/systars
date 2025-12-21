"""
Warp Scheduler for SIMT Streaming Multiprocessor.

The warp scheduler selects which warp to execute each cycle, managing:
- Warp readiness (all operands available)
- Dependency tracking via scoreboard
- Stall detection and handling
- Round-robin or greedy-then-oldest (GTO) scheduling

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      WARP SCHEDULER                              │
    │                                                                  │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │                    WARP TABLE (8 warps)                    │  │
    │  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐       │  │
    │  │  │ W0  │ W1  │ W2  │ W3  │ W4  │ W5  │ W6  │ W7  │       │  │
    │  │  │READY│STALL│EXEC │DONE │READY│STALL│READY│WAIT │       │  │
    │  │  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘       │  │
    │  └─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼───────────┘  │
    │        │     │     │     │     │     │     │     │              │
    │        └─────┴─────┴─────┴─────┴─────┴─────┴─────┘              │
    │                          │                                       │
    │               ┌──────────┴──────────┐                           │
    │               │   SCOREBOARD        │                           │
    │               │  (RAW Hazards)      │                           │
    │               └──────────┬──────────┘                           │
    │                          │                                       │
    │               ┌──────────┴──────────┐                           │
    │               │   ISSUE LOGIC       │                           │
    │               │  (Round-Robin/GTO)  │                           │
    │               └──────────┬──────────┘                           │
    │                          │                                       │
    │                          ▼                                       │
    │                   Selected Warp + Instruction                    │
    └─────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto

from amaranth import Module, Signal, unsigned
from amaranth.lib.wiring import Component, In, Out

from .config import SIMTConfig


class WarpState(IntEnum):
    """State of a warp in the scheduler."""

    INACTIVE = 0  # Warp not allocated
    READY = auto()  # Ready to issue
    ISSUED = auto()  # Instruction issued, waiting for operands
    EXECUTING = auto()  # In execution pipeline
    STALLED_RAW = auto()  # Stalled on read-after-write hazard
    STALLED_BANK = auto()  # Stalled on bank conflict
    STALLED_MEM = auto()  # Stalled on memory access
    STALLED_BARRIER = auto()  # Stalled at barrier
    DONE = auto()  # Completed all instructions


class SchedulingPolicy(IntEnum):
    """Warp scheduling policy."""

    ROUND_ROBIN = 0
    GREEDY_THEN_OLDEST = auto()  # GTO
    LOOSE_ROUND_ROBIN = auto()  # LRR


@dataclass
class Instruction:
    """Represents a SIMT instruction."""

    opcode: str = "NOP"
    dst: int | None = None  # Destination register
    src1: int | None = None  # Source register 1
    src2: int | None = None  # Source register 2
    src3: int | None = None  # Source register 3 (for FMA)
    immediate: int = 0
    latency: int = 1  # Execution latency in cycles

    def __str__(self) -> str:
        """Format instruction for display."""
        if self.opcode == "NOP":
            return "NOP"
        parts = [self.opcode]
        if self.dst is not None:
            parts.append(f"R{self.dst}")
        if self.src1 is not None:
            parts.append(f"R{self.src1}")
        if self.src2 is not None:
            parts.append(f"R{self.src2}")
        if self.src3 is not None:
            parts.append(f"R{self.src3}")
        return " ".join(parts)


@dataclass
class WarpContext:
    """Context for a single warp."""

    warp_id: int
    state: WarpState = WarpState.INACTIVE
    pc: int = 0  # Program counter
    instructions: list = field(default_factory=list)
    current_instruction: Instruction | None = None
    cycles_remaining: int = 0  # Cycles until current instruction completes

    def is_ready(self) -> bool:
        """Check if warp is ready to issue."""
        return self.state == WarpState.READY

    def is_active(self) -> bool:
        """Check if warp is active (not inactive or done)."""
        return self.state not in (WarpState.INACTIVE, WarpState.DONE)

    def has_more_instructions(self) -> bool:
        """Check if warp has more instructions to execute."""
        return self.pc < len(self.instructions)


@dataclass
class Scoreboard:
    """
    Scoreboard for tracking register hazards.

    Tracks which registers have pending writes (in-flight instructions).
    Used to detect RAW (read-after-write) hazards.
    """

    num_registers: int = 256
    pending_writes: set = field(default_factory=set)
    write_latency: dict = field(default_factory=dict)  # reg -> cycles remaining

    def mark_pending(self, reg: int, latency: int) -> None:
        """Mark register as having a pending write."""
        if reg is not None:
            self.pending_writes.add(reg)
            self.write_latency[reg] = latency

    def is_pending(self, reg: int) -> bool:
        """Check if register has pending write."""
        return reg in self.pending_writes

    def check_hazards(self, instr: Instruction) -> bool:
        """Check if instruction has any RAW hazards."""
        for src in [instr.src1, instr.src2, instr.src3]:
            if src is not None and self.is_pending(src):
                return True
        return False

    def tick(self) -> None:
        """Advance one cycle, clearing completed writes."""
        completed = []
        for reg, remaining in self.write_latency.items():
            if remaining <= 1:
                completed.append(reg)
            else:
                self.write_latency[reg] = remaining - 1

        for reg in completed:
            self.pending_writes.discard(reg)
            del self.write_latency[reg]

    def clear(self) -> None:
        """Clear all pending writes."""
        self.pending_writes.clear()
        self.write_latency.clear()


class WarpScheduler(Component):
    """
    RTL warp scheduler component.

    Selects which warp to issue each cycle using round-robin policy.

    Ports:
        # Warp status inputs (from warp contexts)
        warp_ready: Bitmap of ready warps
        warp_stalled: Bitmap of stalled warps

        # Issue interface
        issue_valid: Valid instruction to issue
        issue_warp: Selected warp ID
        issue_pc: Program counter for selected warp

        # Feedback from execution
        exec_complete: Execution completed for warp
        exec_warp: Which warp completed

        # Control
        enable: Enable scheduling
        reset_warps: Reset all warp states
    """

    def __init__(self, config: SIMTConfig, partition_id: int = 0):
        """
        Initialize warp scheduler.

        Args:
            config: SIMT configuration
            partition_id: Identifier for this partition
        """
        self.config = config
        self.partition_id = partition_id
        self.warp_bits = max(1, (config.max_warps_per_partition - 1).bit_length())

        super().__init__(
            {
                # Warp status
                "warp_ready": In(unsigned(config.max_warps_per_partition)),
                "warp_stalled": In(unsigned(config.max_warps_per_partition)),
                # Issue interface
                "issue_valid": Out(1),
                "issue_warp": Out(unsigned(self.warp_bits)),
                # Feedback
                "exec_complete": In(1),
                "exec_warp": In(unsigned(self.warp_bits)),
                # Control
                "enable": In(1),
            }
        )

    def elaborate(self, _platform):
        m = Module()
        cfg = self.config
        num_warps = cfg.max_warps_per_partition

        # Round-robin pointer
        rr_ptr = Signal(unsigned(self.warp_bits), name="rr_ptr")

        # Find next ready warp starting from rr_ptr
        found_ready = Signal(name="found_ready")
        selected_warp = Signal(unsigned(self.warp_bits), name="selected_warp")

        # Priority encoder with rotation
        # Check warps starting from rr_ptr, wrapping around
        for i in range(num_warps):
            warp_idx = Signal(unsigned(self.warp_bits), name=f"warp_idx_{i}")
            m.d.comb += warp_idx.eq((rr_ptr + i) % num_warps)

            warp_ready = Signal(name=f"warp_{i}_ready")
            m.d.comb += warp_ready.eq(self.warp_ready.bit_select(warp_idx, 1))

            # First ready warp wins
            with m.If(warp_ready & ~found_ready):
                m.d.comb += [
                    found_ready.eq(1),
                    selected_warp.eq(warp_idx),
                ]

        # Issue if we found a ready warp and scheduling is enabled
        m.d.comb += [
            self.issue_valid.eq(found_ready & self.enable),
            self.issue_warp.eq(selected_warp),
        ]

        # Update round-robin pointer when we issue
        with m.If(self.issue_valid):
            m.d.sync += rr_ptr.eq((selected_warp + 1) % num_warps)

        return m


# =============================================================================
# Simulation Model for Animation
# =============================================================================


@dataclass
class WarpSchedulerSim:
    """
    Behavioral simulation model of warp scheduler.

    Used for animation and energy estimation without RTL simulation.
    """

    config: SIMTConfig
    partition_id: int = 0
    policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN

    # Warp contexts
    warps: list = field(default_factory=list)

    # Per-warp scoreboards
    scoreboards: list = field(default_factory=list)

    # Round-robin pointer
    rr_ptr: int = 0

    # Statistics
    total_issues: int = 0
    total_stalls: int = 0
    total_energy_pj: float = 0.0

    def __post_init__(self):
        """Initialize warp contexts and scoreboards."""
        self.warps = [WarpContext(warp_id=i) for i in range(self.config.max_warps_per_partition)]
        self.scoreboards = [
            Scoreboard(num_registers=self.config.registers_per_partition)
            for _ in range(self.config.max_warps_per_partition)
        ]

    def load_program(self, warp_id: int, instructions: list[Instruction]) -> None:
        """Load program into a warp."""
        warp = self.warps[warp_id]
        warp.instructions = instructions
        warp.pc = 0
        warp.state = WarpState.READY if instructions else WarpState.DONE

    def activate_warps(self, num_warps: int) -> None:
        """Activate a number of warps with their programs."""
        for i in range(min(num_warps, len(self.warps))):
            if self.warps[i].has_more_instructions():
                self.warps[i].state = WarpState.READY

    def _find_ready_warp_rr(self) -> int | None:
        """Find next ready warp using round-robin."""
        num_warps = len(self.warps)
        for i in range(num_warps):
            warp_idx = (self.rr_ptr + i) % num_warps
            warp = self.warps[warp_idx]
            if warp.is_ready() and warp.has_more_instructions():
                # Check for hazards
                instr = warp.instructions[warp.pc]
                if not self.scoreboards[warp_idx].check_hazards(instr):
                    return warp_idx
        return None

    def schedule(self) -> tuple[int, Instruction] | None:
        """
        Select a warp and instruction to issue.

        Returns:
            Tuple of (warp_id, instruction) or None if all stalled.
        """
        # Update scoreboards
        for sb in self.scoreboards:
            sb.tick()

        # Find ready warp
        if self.policy == SchedulingPolicy.ROUND_ROBIN:
            warp_idx = self._find_ready_warp_rr()
        else:
            warp_idx = self._find_ready_warp_rr()  # Default to RR

        if warp_idx is None:
            self.total_stalls += 1
            return None

        warp = self.warps[warp_idx]
        instr = warp.instructions[warp.pc]

        # Update state
        warp.pc += 1
        warp.current_instruction = instr
        warp.state = WarpState.EXECUTING
        warp.cycles_remaining = instr.latency

        # Update scoreboard
        if instr.dst is not None:
            self.scoreboards[warp_idx].mark_pending(instr.dst, instr.latency)

        # Update round-robin pointer
        self.rr_ptr = (warp_idx + 1) % len(self.warps)

        # Statistics
        self.total_issues += 1
        self.total_energy_pj += self.config.scheduler_energy_pj

        return warp_idx, instr

    def tick(self) -> None:
        """Advance one cycle for all warps."""
        for warp in self.warps:
            if warp.state == WarpState.EXECUTING:
                warp.cycles_remaining -= 1
                if warp.cycles_remaining <= 0:
                    # Execution complete
                    if warp.has_more_instructions():
                        warp.state = WarpState.READY
                    else:
                        warp.state = WarpState.DONE
                    warp.current_instruction = None

    def get_warp_states(self) -> list[WarpState]:
        """Get states of all warps."""
        return [warp.state for warp in self.warps]

    def get_active_count(self) -> int:
        """Get number of active warps."""
        return sum(1 for warp in self.warps if warp.is_active())

    def get_ready_count(self) -> int:
        """Get number of ready warps."""
        return sum(1 for warp in self.warps if warp.is_ready())

    def all_done(self) -> bool:
        """Check if all warps are done."""
        return all(warp.state in (WarpState.INACTIVE, WarpState.DONE) for warp in self.warps)

    def get_visualization(self) -> str:
        """Get ASCII visualization of warp states."""
        state_symbols = {
            WarpState.INACTIVE: "·",
            WarpState.READY: "R",
            WarpState.ISSUED: "I",
            WarpState.EXECUTING: "E",
            WarpState.STALLED_RAW: "W",
            WarpState.STALLED_BANK: "B",
            WarpState.STALLED_MEM: "M",
            WarpState.STALLED_BARRIER: "S",
            WarpState.DONE: "D",
        }
        return "".join(state_symbols[warp.state] for warp in self.warps)
