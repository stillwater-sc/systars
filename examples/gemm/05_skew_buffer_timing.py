#!/usr/bin/env python3
"""
Skew Buffer and SRAM Timing Visualization.

This example demonstrates the timing relationship between:
1. Banked SRAM reads with multi-cycle latency
2. Skew buffers that stagger data for systolic wavefront injection
3. The resulting wavefront pattern in the systolic array

Memory Layout (after DMA preparation):
    Scratchpad A (column-major / transposed):
        Address k contains: [A[0,k], A[1,k], ..., A[N-1,k]] (column k of A)

    Scratchpad B (row-major):
        Address k contains: [B[k,0], B[k,1], ..., B[k,M-1]] (row k of B)

    Note: DMA must transpose A when loading from DRAM (row-major) to scratchpad

Timing (250MHz, 4-cycle SRAM latency):

    Cycle:  0   1   2   3   4   5   6   7   8   9  10  11  12  ...
            │   │   │   │   │   │   │   │   │   │   │   │   │
    SRAM:   ├───────────────┤
            │ Read k=0      │ Data valid
            │               ├───────────────┤
            │               │ Read k=1      │ Data valid
            │               │               ├───────────────┤
            │               │               │ Read k=2      │ ...

    Skew Buffer Output (after SRAM latency):
                            Lane 0: ──[k=0]──[k=1]──[k=2]──...
                            Lane 1: ────────[k=0]──[k=1]──[k=2]──...
                            Lane 2: ──────────────[k=0]──[k=1]──...
                            Lane 3: ────────────────────[k=0]──...

    Array PE[i,j] first valid input at cycle: SRAM_latency + max(i,j)

Usage:
    python 05_skew_buffer_timing.py [--size N] [--latency L] [--k K]
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# Add parent directory to path for examples.common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.cli import add_animation_args, add_gemm_args, get_effective_delay


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# Memory and Buffer Models
# =============================================================================


@dataclass
class SRAMBank:
    """Model of an SRAM bank with multi-cycle latency."""

    latency: int = 4
    data: dict = field(default_factory=dict)

    # Pipeline state
    pending_reads: list = field(default_factory=list)
    output_data: any = None
    output_valid: bool = False

    def write(self, addr: int, data: np.ndarray):
        """Write data to address (instant for simulation)."""
        self.data[addr] = data.copy()

    def request_read(self, addr: int):
        """Issue a read request (will be valid after latency cycles)."""
        self.pending_reads.append({"addr": addr, "cycles_remaining": self.latency})

    def tick(self):
        """Advance one clock cycle."""
        self.output_valid = False
        self.output_data = None

        # Process pending reads
        new_pending = []
        for req in self.pending_reads:
            req["cycles_remaining"] -= 1
            if req["cycles_remaining"] <= 0:
                # Read completes
                addr = req["addr"]
                self.output_data = self.data.get(addr, None)
                self.output_valid = True
            else:
                new_pending.append(req)
        self.pending_reads = new_pending


@dataclass
class SkewBuffer:
    """Model of skew buffer with per-lane delay.

    Tracks data values, validity, and control signals (SOS/EOS) through the delay.
    """

    num_lanes: int
    # Each lane has a FIFO of depth = lane index
    fifos: list = field(default_factory=list)
    valid_fifos: list = field(default_factory=list)
    sos_fifos: list = field(default_factory=list)  # Start-of-stream signals
    eos_fifos: list = field(default_factory=list)  # End-of-stream signals

    def __post_init__(self):
        # Lane i has i stages of delay
        self.fifos = [[] for _ in range(self.num_lanes)]
        self.valid_fifos = [[] for _ in range(self.num_lanes)]
        self.sos_fifos = [[] for _ in range(self.num_lanes)]
        self.eos_fifos = [[] for _ in range(self.num_lanes)]

    def push(self, lane: int, data: int, valid: bool = True, sos: bool = False, eos: bool = False):
        """Push data and control signals into a lane's input."""
        if lane < self.num_lanes:
            self.fifos[lane].append(data)
            self.valid_fifos[lane].append(valid)
            self.sos_fifos[lane].append(sos)
            self.eos_fifos[lane].append(eos)

    def tick(self) -> tuple:
        """Advance one cycle, return (outputs, sos_list, eos_list) for each lane.

        Returns:
            outputs: list of (value, valid) tuples
            sos_list: list of bool
            eos_list: list of bool
        """
        outputs = []
        sos_list = []
        eos_list = []

        for lane in range(self.num_lanes):
            depth = lane  # Lane i has i stages of delay

            if depth == 0:
                # Lane 0: direct pass-through (output what was pushed this cycle)
                if self.fifos[lane]:
                    out = self.fifos[lane].pop(0)
                    valid = self.valid_fifos[lane].pop(0)
                    sos = self.sos_fifos[lane].pop(0)
                    eos = self.eos_fifos[lane].pop(0)
                    outputs.append((out, valid))
                    sos_list.append(sos)
                    eos_list.append(eos)
                else:
                    outputs.append((None, False))
                    sos_list.append(False)
                    eos_list.append(False)
            else:
                # Lane i: output oldest item if FIFO has MORE than depth items
                # Item pushed at cycle t outputs at cycle t+depth
                if len(self.fifos[lane]) > depth:
                    out = self.fifos[lane].pop(0)
                    valid = self.valid_fifos[lane].pop(0)
                    sos = self.sos_fifos[lane].pop(0)
                    eos = self.eos_fifos[lane].pop(0)
                    outputs.append((out, valid))
                    sos_list.append(sos)
                    eos_list.append(eos)
                else:
                    outputs.append((None, False))
                    sos_list.append(False)
                    eos_list.append(False)

        return outputs, sos_list, eos_list

    def push_all(self, data: list, valid: bool = True, sos: bool = False, eos: bool = False):
        """Push data and control signals to all lanes simultaneously."""
        for lane, d in enumerate(data):
            self.push(lane, d, valid, sos, eos)

    def get_register_state(self) -> list:
        """Return the current state of all shift registers for visualization.

        Returns:
            List of lane states, where each lane state is a list of
            (value, valid, sos, eos) tuples representing register contents.
            Lane i has i register stages (lane 0 is pass-through with 0 stages).
            The list is ordered from oldest (output side) to newest (input side).
        """
        state = []
        for lane in range(self.num_lanes):
            lane_regs = []
            for idx in range(len(self.fifos[lane])):
                lane_regs.append(
                    (
                        self.fifos[lane][idx],
                        self.valid_fifos[lane][idx],
                        self.sos_fifos[lane][idx],
                        self.eos_fifos[lane][idx],
                    )
                )
            state.append(lane_regs)
        return state


@dataclass
class PE:
    """Processing Element with input registers, MAC, output ports, and control signals.

    In output-stationary dataflow:
    - a_in flows from left neighbor (or skew buffer for column 0)
    - b_in flows from top neighbor (or skew buffer for row 0)
    - c accumulates locally: c += a * b
    - a_out = a_in (passed to right neighbor)
    - b_out = b_in (passed to bottom neighbor)

    Control signals (flow with data):
    - sos (start-of-stream): Reset accumulator, marks first data of a tile
    - eos (end-of-stream): Marks last data of a tile, can trigger result output
    """

    # Input register values (set by neighbors or skew buffer)
    a_in: int = 0
    b_in: int = 0
    a_valid: bool = False
    b_valid: bool = False

    # Control signals (input)
    a_sos: bool = False  # Start-of-stream from left
    a_eos: bool = False  # End-of-stream from left
    b_sos: bool = False  # Start-of-stream from top
    b_eos: bool = False  # End-of-stream from top

    # Accumulator
    c: int = 0

    # Local state for visualization
    active: bool = False  # Currently computing
    done: bool = False  # Received EOS, computation complete

    # Output ports (computed on clock edge)
    a_out: int = 0
    b_out: int = 0
    a_out_valid: bool = False
    b_out_valid: bool = False

    # Control signals (output, propagate to neighbors)
    a_out_sos: bool = False
    a_out_eos: bool = False
    b_out_sos: bool = False
    b_out_eos: bool = False

    def tick(self):
        """Systolic clock edge: compute MAC and pass data through."""
        # Reset accumulator on start-of-stream
        if (self.a_sos and self.a_valid) or (self.b_sos and self.b_valid):
            self.c = 0
            self.done = False

        # MAC when both inputs are valid
        if self.a_valid and self.b_valid:
            self.c += self.a_in * self.b_in
            self.active = True
        else:
            self.active = False

        # Mark done on end-of-stream (when both A and B have finished)
        if self.a_eos and self.a_valid and self.b_eos and self.b_valid:
            self.done = True

        # Pass data to outputs (flows to neighbors)
        self.a_out = self.a_in
        self.b_out = self.b_in
        self.a_out_valid = self.a_valid
        self.b_out_valid = self.b_valid

        # Pass control signals to outputs
        self.a_out_sos = self.a_sos
        self.a_out_eos = self.a_eos
        self.b_out_sos = self.b_sos
        self.b_out_eos = self.b_eos


@dataclass
class SystolicArray:
    """Grid of PEs with proper systolic interconnections.

    Data flow:
    - A values enter from left edge (row i gets skew buffer lane i)
    - B values enter from top edge (column j gets skew buffer lane j)
    - A flows left→right through each row
    - B flows top→bottom through each column

    Control signals:
    - sos/eos flow with data to mark stream boundaries
    """

    size: int
    pes: list = field(default_factory=list)  # 2D array of PEs

    def __post_init__(self):
        self.pes = [[PE() for _ in range(self.size)] for _ in range(self.size)]

    def set_inputs(
        self,
        a_edge: list,
        b_edge: list,
        a_sos: list = None,
        a_eos: list = None,
        b_sos: list = None,
        b_eos: list = None,
    ):
        """Set inputs from skew buffers to array edges.

        a_edge: list of (value, valid) tuples for left column (rows 0..N-1)
        b_edge: list of (value, valid) tuples for top row (columns 0..N-1)
        a_sos/a_eos: list of bool for control signals (same length as a_edge)
        b_sos/b_eos: list of bool for control signals (same length as b_edge)
        """
        # Default control signals to False
        if a_sos is None:
            a_sos = [False] * self.size
        if a_eos is None:
            a_eos = [False] * self.size
        if b_sos is None:
            b_sos = [False] * self.size
        if b_eos is None:
            b_eos = [False] * self.size

        # A feeds into left column PE[i,0]
        for i in range(self.size):
            val, valid = a_edge[i]
            self.pes[i][0].a_in = val if valid else 0
            self.pes[i][0].a_valid = valid
            self.pes[i][0].a_sos = a_sos[i]
            self.pes[i][0].a_eos = a_eos[i]

        # B feeds into top row PE[0,j]
        for j in range(self.size):
            val, valid = b_edge[j]
            self.pes[0][j].b_in = val if valid else 0
            self.pes[0][j].b_valid = valid
            self.pes[0][j].b_sos = b_sos[j]
            self.pes[0][j].b_eos = b_eos[j]

        # Interior PEs get inputs from neighbors
        # A from left neighbor (same row, previous column)
        for i in range(self.size):
            for j in range(1, self.size):
                self.pes[i][j].a_in = self.pes[i][j - 1].a_out
                self.pes[i][j].a_valid = self.pes[i][j - 1].a_out_valid
                self.pes[i][j].a_sos = self.pes[i][j - 1].a_out_sos
                self.pes[i][j].a_eos = self.pes[i][j - 1].a_out_eos

        # B from top neighbor (previous row, same column)
        for i in range(1, self.size):
            for j in range(self.size):
                self.pes[i][j].b_in = self.pes[i - 1][j].b_out
                self.pes[i][j].b_valid = self.pes[i - 1][j].b_out_valid
                self.pes[i][j].b_sos = self.pes[i - 1][j].b_out_sos
                self.pes[i][j].b_eos = self.pes[i - 1][j].b_out_eos

    def tick(self):
        """Clock all PEs."""
        for i in range(self.size):
            for j in range(self.size):
                self.pes[i][j].tick()

    def all_done(self) -> bool:
        """Check if all PEs have received EOS and completed."""
        for i in range(self.size):
            for j in range(self.size):
                if not self.pes[i][j].done:
                    return False
        return True

    def get_accumulator(self) -> np.ndarray:
        """Return the accumulator values as a numpy array."""
        result = np.zeros((self.size, self.size), dtype=np.int32)
        for i in range(self.size):
            for j in range(self.size):
                result[i, j] = self.pes[i][j].c
        return result

    def get_state_grid(self) -> list:
        """Return grid of PE states for visualization.

        Returns list of lists with state chars:
        - '·' = idle (no valid inputs yet)
        - '*' = active (computing)
        - '✓' = done (received EOS)
        """
        grid = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                pe = self.pes[i][j]
                if pe.done:
                    row.append("✓")
                elif pe.active:
                    row.append("*")
                else:
                    row.append("·")
            grid.append(row)
        return grid


@dataclass
class SkewedFeeder:
    """Complete model of SRAM + skew buffer system."""

    size: int
    sram_latency: int
    k_dim: int

    sram_a: SRAMBank = None
    sram_b: SRAMBank = None
    skew_a: SkewBuffer = None
    skew_b: SkewBuffer = None

    cycle: int = 0
    k_issued: int = 0
    feeding_done: bool = False

    # State tracking for visualization
    sram_a_req_addr: int = -1
    sram_b_req_addr: int = -1

    def __post_init__(self):
        self.sram_a = SRAMBank(latency=self.sram_latency)
        self.sram_b = SRAMBank(latency=self.sram_latency)
        self.skew_a = SkewBuffer(self.size)
        self.skew_b = SkewBuffer(self.size)

    def load_matrices(self, A: np.ndarray, B: np.ndarray):
        """
        Load matrices into SRAM banks.

        A is stored column-major (transposed):
            Address k contains column k of A = [A[0,k], A[1,k], ...]

        B is stored row-major:
            Address k contains row k of B = [B[k,0], B[k,1], ...]
        """
        # Store A transposed: address k = column k of A
        for k in range(A.shape[1]):
            col_k = A[:, k].copy()  # Column k of A
            self.sram_a.write(k, col_k)

        # Store B directly: address k = row k of B
        for k in range(B.shape[0]):
            row_k = B[k, :].copy()  # Row k of B
            self.sram_b.write(k, row_k)

    def step(self):
        """Execute one clock cycle.

        Returns:
            a_outputs: list of (value, valid) tuples for A lanes
            b_outputs: list of (value, valid) tuples for B lanes
            a_sos: list of bool - start-of-stream for A lanes
            a_eos: list of bool - end-of-stream for A lanes
            b_sos: list of bool - start-of-stream for B lanes
            b_eos: list of bool - end-of-stream for B lanes
        """
        self.sram_a_req_addr = -1
        self.sram_b_req_addr = -1

        # Track which k value's data will complete in this cycle's SRAM tick.
        # Data requested at cycle C completes after `latency` ticks (at cycle C + latency - 1).
        # So data completing at cycle N was requested at cycle (N - latency + 1).
        k_completing = self.cycle - self.sram_latency + 1
        if k_completing < 0 or k_completing >= self.k_dim:
            k_completing = -1  # No valid completion this cycle

        # Issue new read requests if not done
        if self.k_issued < self.k_dim:
            self.sram_a.request_read(self.k_issued)
            self.sram_b.request_read(self.k_issued)
            self.sram_a_req_addr = self.k_issued
            self.sram_b_req_addr = self.k_issued
            self.k_issued += 1

        # Tick SRAM banks
        self.sram_a.tick()
        self.sram_b.tick()

        # Feed SRAM outputs to skew buffers with SOS/EOS signals
        if self.sram_a.output_valid and self.sram_b.output_valid:
            # k_completing tells us which k value's data just became valid
            is_sos = k_completing == 0  # First k value
            is_eos = k_completing == self.k_dim - 1  # Last k value

            self.skew_a.push_all(list(self.sram_a.output_data), valid=True, sos=is_sos, eos=is_eos)
            self.skew_b.push_all(list(self.sram_b.output_data), valid=True, sos=is_sos, eos=is_eos)
        else:
            # Push invalid data to keep lanes synchronized
            self.skew_a.push_all([0] * self.size, valid=False, sos=False, eos=False)
            self.skew_b.push_all([0] * self.size, valid=False, sos=False, eos=False)

        # Tick skew buffers and get outputs with control signals
        a_outputs, a_sos, a_eos = self.skew_a.tick()
        b_outputs, b_sos, b_eos = self.skew_b.tick()

        self.cycle += 1

        # feeding_done is now determined by the array's all_done() method
        # but we still track when the feeder has finished outputting
        last_lane_delay = self.size - 1
        expected_end = self.sram_latency + self.k_dim + last_lane_delay
        if self.cycle >= expected_end:
            self.feeding_done = True

        return a_outputs, b_outputs, a_sos, a_eos, b_sos, b_eos


# =============================================================================
# Visualization
# =============================================================================


def draw_timing_diagram(
    size: int,
    sram_latency: int,
    k_dim: int,
    use_color: bool = True,
):
    """Draw a text-based timing diagram of the skew buffer operation."""
    if not use_color:
        Colors.disable()
    c = Colors

    print(f"\n{c.BOLD}╔════════════════════════════════════════════════════════════════════════╗")
    print("║              SRAM + SKEW BUFFER TIMING DIAGRAM                         ║")
    print(f"╚════════════════════════════════════════════════════════════════════════╝{c.RESET}\n")

    print(f"Configuration: {size}x{size} array, {sram_latency}-cycle SRAM, K={k_dim}\n")

    total_cycles = sram_latency + k_dim + size + 2
    col_width = 4

    # Header - cycle numbers
    header = "Cycle:     "
    for cyc in range(total_cycles):
        header += f"{cyc:^{col_width}}"
    print(header)
    print("           " + "─" * (total_cycles * col_width))

    # SRAM read requests
    sram_row = "SRAM req:  "
    for cyc in range(total_cycles):
        if cyc < k_dim:
            sram_row += f"{c.CYAN}k={cyc:<2}{c.RESET}"
        else:
            sram_row += " " * col_width
    print(sram_row)

    # SRAM data valid
    valid_row = "SRAM valid:"
    for cyc in range(total_cycles):
        k_completing = cyc - sram_latency
        if 0 <= k_completing < k_dim:
            valid_row += f"{c.GREEN}k={k_completing:<2}{c.RESET}"
        else:
            valid_row += " " * col_width
    print(valid_row)

    print()
    print(f"{c.BOLD}Skew Buffer Outputs (A lanes = rows, B lanes = columns):{c.RESET}")
    print()

    # Skew buffer outputs for each lane
    for lane in range(size):
        lane_row = f"Lane {lane}:    "
        skew_delay = lane

        for cyc in range(total_cycles):
            # Data arrives at skew input at cycle (sram_latency + k)
            # Data exits lane at cycle (sram_latency + k + lane_delay)
            k_output = cyc - sram_latency - skew_delay
            if 0 <= k_output < k_dim:
                if lane == 0:
                    lane_row += f"{c.YELLOW}k={k_output:<2}{c.RESET}"
                else:
                    lane_row += f"{c.MAGENTA}k={k_output:<2}{c.RESET}"
            else:
                lane_row += "  · "
        print(lane_row)

    print()

    # Show when each PE gets first valid input
    print(f"{c.BOLD}First valid input at each PE location:{c.RESET}")
    print("(PE[row,col] receives first (A,B) pair at cycle)")
    print()

    cell_width = 4
    header = "     "
    for col in range(size):
        header += f"c{col:^{cell_width - 1}}"
    print(header)

    for row in range(size):
        row_str = f"r{row}   "
        for col in range(size):
            # First valid input when both A[row] and B[col] are valid
            # A[row] valid at: sram_latency + row (first k=0)
            # B[col] valid at: sram_latency + col (first k=0)
            first_valid = sram_latency + max(row, col)
            row_str += f"{first_valid:^{cell_width}}"
        print(row_str)

    print()

    # Total latency breakdown
    print(f"{c.BOLD}Latency Breakdown:{c.RESET}")
    print(f"  SRAM read latency:     {sram_latency} cycles")
    print(f"  Max skew delay:        {size - 1} cycles (lane {size - 1})")
    print(f"  First output (lane 0): cycle {sram_latency}")
    print(f"  Last output (lane {size - 1}):  cycle {sram_latency + k_dim - 1 + size - 1}")
    print(f"  Total feeding window:  {k_dim + size - 1} cycles")
    print()


def draw_memory_layout(size: int, use_color: bool = True):
    """Show the expected memory layout after DMA preparation."""
    if not use_color:
        Colors.disable()
    c = Colors

    print(f"\n{c.BOLD}╔════════════════════════════════════════════════════════════════════════╗")
    print("║                    MEMORY LAYOUT FOR MATMUL                            ║")
    print(f"╚════════════════════════════════════════════════════════════════════════╝{c.RESET}\n")

    print(f"{c.BOLD}Goal:{c.RESET} Compute C = A @ B where A is {size}×K and B is K×{size}")
    print()

    print(f"{c.BOLD}DRAM Layout (row-major, standard):{c.RESET}")
    print("  A in DRAM: A[i,k] at address i*K + k")
    print(f"  B in DRAM: B[k,j] at address k*{size} + j")
    print()

    print(f"{c.BOLD}Scratchpad Layout (after DMA transpose):{c.RESET}")
    print()
    print(f"  {c.YELLOW}Scratchpad A Bank (column-major / transposed):{c.RESET}")
    print("  ┌────────────────────────────────────────────┐")
    print("  │ Address k contains column k of A:         │")
    print("  │   [A[0,k], A[1,k], A[2,k], ..., A[N-1,k]] │")
    print("  └────────────────────────────────────────────┘")
    print()
    print("  Example for K=3:")
    print("    Addr 0: [A[0,0], A[1,0], A[2,0], A[3,0]]  ← column 0")
    print("    Addr 1: [A[0,1], A[1,1], A[2,1], A[3,1]]  ← column 1")
    print("    Addr 2: [A[0,2], A[1,2], A[2,2], A[3,2]]  ← column 2")
    print()
    print(f"  {c.CYAN}Scratchpad B Bank (row-major):{c.RESET}")
    print("  ┌────────────────────────────────────────────┐")
    print("  │ Address k contains row k of B:            │")
    print("  │   [B[k,0], B[k,1], B[k,2], ..., B[k,M-1]] │")
    print("  └────────────────────────────────────────────┘")
    print()
    print("  Example for K=3:")
    print("    Addr 0: [B[0,0], B[0,1], B[0,2], B[0,3]]  ← row 0")
    print("    Addr 1: [B[1,0], B[1,1], B[1,2], B[1,3]]  ← row 1")
    print("    Addr 2: [B[2,0], B[2,1], B[2,2], B[2,3]]  ← row 2")
    print()

    print(f"{c.BOLD}DMA Operations:{c.RESET}")
    print("  1. Load A: DRAM (row-major) → Scratchpad A (column-major)")
    print(f"     {c.RED}⚠ Requires transpose during DMA transfer{c.RESET}")
    print("     Option 1: DMA hardware performs transpose (strided access)")
    print("     Option 2: CPU pre-transposes A in DRAM before DMA")
    print()
    print("  2. Load B: DRAM (row-major) → Scratchpad B (row-major)")
    print(f"     {c.GREEN}✓ Direct copy, no transformation needed{c.RESET}")
    print()

    print(f"{c.BOLD}Why this layout?{c.RESET}")
    print("  For matmul iteration k, we need:")
    print("    - Column k of A: feeds rows of systolic array (A[i,k] → row i)")
    print("    - Row k of B: feeds cols of systolic array (B[k,j] → col j)")
    print()
    print("  With this layout, a single SRAM read at address k provides:")
    print("    - All elements needed for array rows (from A bank)")
    print("    - All elements needed for array cols (from B bank)")
    print()


def animate_feeding(
    size: int = 4,
    sram_latency: int = 4,
    k_dim: int = 4,
    delay_ms: int = 500,
    use_color: bool = True,
    movie_mode: bool = False,
    step: bool = False,
):
    """Animate the feeding process showing SRAM reads and skew buffer state."""
    if not use_color:
        Colors.disable()
    c = Colors

    # Create test matrices
    np.random.seed(42)
    A = np.random.randint(1, 5, size=(size, k_dim), dtype=np.int8)
    B = np.random.randint(1, 5, size=(k_dim, size), dtype=np.int8)

    # Initialize feeder
    feeder = SkewedFeeder(size=size, sram_latency=sram_latency, k_dim=k_dim)
    feeder.load_matrices(A, B)

    print(f"\n{c.BOLD}Input Matrices:{c.RESET}")
    print(f"\nA ({size}x{k_dim}):")
    print(A)
    print(f"\nB ({k_dim}x{size}):")
    print(B)
    print("\nExpected C = A @ B:")
    print(A @ B)

    # Prompt before starting (skip in movie mode for term2svg capture)
    if not movie_mode and sys.stdin.isatty():
        if step:
            print(f"\n{c.DIM}Step mode: Press Enter to advance each cycle...{c.RESET}")
        else:
            print(f"\n{c.DIM}Press Enter to start animation...{c.RESET}")
        input()
    elif not movie_mode:
        print(f"\n{c.DIM}(Non-interactive mode, starting immediately...){c.RESET}")

    # Need enough cycles for data to fully propagate through the array
    # sram_latency + k_dim + skew_delay + array_propagation
    max_cycles = sram_latency + k_dim + (size - 1) + (size - 1) + 2

    # Create systolic array with hierarchical PE structure
    sa = SystolicArray(size=size)

    # Timeline log for post-animation summary
    timeline_log = []

    # Track PE state changes for logging
    prev_active_pes = set()
    prev_done_pes = set()

    for cycle_num in range(max_cycles):
        # Execute cycle and get skew buffer outputs with control signals
        a_outputs, b_outputs, a_sos, a_eos, b_sos, b_eos = feeder.step()

        # Get skew buffer state AFTER step (shows registers waiting to output)
        a_regs = feeder.skew_a.get_register_state()
        b_regs = feeder.skew_b.get_register_state()

        # ─────────────────────────────────────────────────────────────────
        # TIMELINE LOGGING (accumulated for post-animation display)
        # ─────────────────────────────────────────────────────────────────
        cycle_events = []

        # Log SRAM requests
        if feeder.sram_a_req_addr >= 0:
            cycle_events.append(f"SRAM: request k={feeder.sram_a_req_addr}")

        # Log SRAM data valid
        if feeder.sram_a.output_valid:
            k_val = cycle_num - sram_latency + 1
            cycle_events.append(f"SRAM: data k={k_val} valid → SkewBuffer input")

        # Log skew buffer outputs
        a_valid_lanes = [i for i in range(size) if a_outputs[i][1]]
        b_valid_lanes = [i for i in range(size) if b_outputs[i][1]]
        if a_valid_lanes:
            a_sos_lanes = [i for i in a_valid_lanes if a_sos[i]]
            a_eos_lanes = [i for i in a_valid_lanes if a_eos[i]]
            sos_str = f" SOS:{a_sos_lanes}" if a_sos_lanes else ""
            eos_str = f" EOS:{a_eos_lanes}" if a_eos_lanes else ""
            cycle_events.append(f"SkewA: lanes {a_valid_lanes} → rows{sos_str}{eos_str}")
        if b_valid_lanes:
            b_sos_lanes = [i for i in b_valid_lanes if b_sos[i]]
            b_eos_lanes = [i for i in b_valid_lanes if b_eos[i]]
            sos_str = f" SOS:{b_sos_lanes}" if b_sos_lanes else ""
            eos_str = f" EOS:{b_eos_lanes}" if b_eos_lanes else ""
            cycle_events.append(f"SkewB: lanes {b_valid_lanes} → cols{sos_str}{eos_str}")

        # Now display everything - all state is from AFTER this cycle's processing
        clear_screen()

        print(f"{c.BOLD}╔════════════════════════════════════════════════════════════════╗")
        print(f"║  SKEW BUFFER ANIMATION  -  Cycle {cycle_num:<3}                           ║")
        print(f"╚════════════════════════════════════════════════════════════════╝{c.RESET}")
        print()

        # Show SRAM state (now consistent with skew buffer state)
        print(f"{c.BOLD}SRAM Banks:{c.RESET}")
        if feeder.sram_a_req_addr >= 0:
            print(f"  Read request: k={feeder.sram_a_req_addr}")
        else:
            print(f"  {c.DIM}Read request: --{c.RESET}")

        if feeder.sram_a.output_valid:
            # k_val = cycle_num - sram_latency + 1 (data requested at that cycle completes now)
            k_val = cycle_num - sram_latency + 1
            a_data = [int(x) for x in feeder.sram_a.output_data]
            b_data = [int(x) for x in feeder.sram_b.output_data]
            print(f"  {c.GREEN}Data valid:   k={k_val}{c.RESET}  A={a_data}  B={b_data}")
        else:
            print(f"  {c.DIM}Data valid:   (waiting...){c.RESET}")

        # Feed skew buffer outputs to array edges with control signals, then clock all PEs
        sa.set_inputs(a_outputs, b_outputs, a_sos, a_eos, b_sos, b_eos)
        sa.tick()

        # Get array state for merged display
        array_c = sa.get_accumulator()
        state_grid = sa.get_state_grid()

        # Log PE state changes
        curr_active_pes = set()
        curr_done_pes = set()
        for i in range(size):
            for j in range(size):
                if state_grid[i][j] == "*":
                    curr_active_pes.add((i, j))
                if sa.pes[i][j].done:
                    curr_done_pes.add((i, j))

        # Log newly activated PEs
        new_active = curr_active_pes - prev_active_pes
        if new_active:
            pe_list = [f"PE[{i},{j}]" for i, j in sorted(new_active)]
            cycle_events.append(f"Array: {', '.join(pe_list)} computing")

        # Log newly completed PEs
        new_done = curr_done_pes - prev_done_pes
        if new_done:
            pe_list = [f"PE[{i},{j}]={array_c[i, j]}" for i, j in sorted(new_done)]
            cycle_events.append(f"Array: {', '.join(pe_list)} done (EOS)")

        prev_active_pes = curr_active_pes
        prev_done_pes = curr_done_pes

        # Add this cycle's events to the timeline
        if cycle_events:
            timeline_log.append((cycle_num, cycle_events))

        # =================================================================
        # 2D PHYSICAL LAYOUT VISUALIZATION
        # =================================================================
        # Layout:
        #   - B Scratchpad (top) → Skew Buffer B (horizontal) → Array columns
        #   - A Scratchpad (left) → Skew Buffer A (vertical) → Array rows
        #   - Systolic Array (center) with merged PE state + accumulator

        max_depth = size - 1

        def fmt_val(val, valid, sos, eos, color):
            """Format a skew buffer value as exactly 4 visible chars (caller adds 1 space)."""
            if valid:
                mark = "S" if sos else ("E" if eos else " ")
                mark_c = c.GREEN if sos else (c.RED if eos else "")
                mark_r = c.RESET if mark != " " else ""
                # " {val:>2}{mark}" = 4 visible chars, e.g. "  4S" (ones digit at position 3)
                return f" {color}{val:>2}{c.RESET}{mark_c}{mark}{mark_r}"
            else:
                # "  · " = 4 visible chars, dot at position 3 to align with ones digit
                return f"{c.DIM}  · {c.RESET}"

        # Helper to format PE cell (value + state)
        def fmt_pe(val, state, done):
            if done:
                return f"{c.GREEN}{val:>3}✓{c.RESET}"
            elif state == "*":
                return f"{c.YELLOW}{val:>3}*{c.RESET}"
            else:
                return f"{c.DIM}  · {c.RESET}"

        # Calculate left margin for centering B section above array
        # Calculate left margin dynamically to align B columns with array columns
        # Skew A prefix width = a_sram_cell(8) + skew_cells(max_depth*4) + out_cell(4) + " →r{row}│ "(6)
        # B section adds 11 chars after left_margin, so: left_margin + 11 = skew_a_prefix
        left_margin = 7 + max_depth * 4

        # ─────────────────────────────────────────────────────────────────
        # B SCRATCHPAD BANK (top, horizontal)
        # ─────────────────────────────────────────────────────────────────
        print()
        print(" " * left_margin + f"{c.BOLD}B Scratchpad Bank{c.RESET}")
        if feeder.sram_b.output_valid:
            b_data = [int(x) for x in feeder.sram_b.output_data]
            b_str = "  ".join(f"{c.CYAN}{v:>2}{c.RESET}" for v in b_data)
            print(" " * left_margin + f"  SRAM out: [{b_str}]")
        else:
            print(" " * left_margin + f"  {c.DIM}SRAM out: (waiting...){c.RESET}")

        # ─────────────────────────────────────────────────────────────────
        # SKEW BUFFER B (horizontal, delays for columns)
        # Data flows: SRAM → In → R(N-1) → ... → R0 → Out → Array
        # So display top-to-bottom: In, R(max-1), ..., R0, Out, ↓
        # ─────────────────────────────────────────────────────────────────
        print(" " * left_margin + f"{c.BOLD}Skew Buffer B{c.RESET} (SRAM → delays → columns)")

        # Column headers
        col_header = " " * left_margin + "           "
        for col in range(size):
            col_header += f" b{col}  "
        print(col_header)

        # Delay registers from deepest (newest) to shallowest (oldest/next to output)
        # R(max-1) is deepest (input side), R0 is next to output
        # FIFO fills from input side: when len < depth, items are at R(depth-len) through R(depth-1)
        # Formula: fifo[i] is at R[depth - len + i], so R[d] corresponds to fifo[d - depth + len]
        for d in range(max_depth - 1, -1, -1):
            reg_row = " " * left_margin + f"    R{d}     "
            for col in range(size):
                depth = col  # Column c has c delay stages
                regs = b_regs[col]
                fifo_len = len(regs)
                fifo_idx = d - depth + fifo_len  # Map R[d] to FIFO index
                if d < depth and 0 <= fifo_idx < fifo_len:
                    val, valid, sos, eos = regs[fifo_idx]
                    reg_row += fmt_val(val, valid, sos, eos, c.CYAN) + " "  # 4+1=5 chars
                elif d < depth:
                    reg_row += f"{c.DIM}  · {c.RESET} "  # 4+1=5 chars, dot at position 3
                else:
                    reg_row += "     "  # 5 spaces for unused positions
            print(reg_row)

        # Output row (what's being fed to array this cycle) - at bottom, near array
        out_row = " " * left_margin + "   Out →   "
        for col in range(size):
            b_val, b_val_valid = b_outputs[col]
            out_row += (
                fmt_val(b_val, b_val_valid, b_sos[col], b_eos[col], c.CYAN) + " "
            )  # 4+1=5 chars
        print(out_row)

        # Arrows showing data flow down into array
        arrow_row = " " * left_margin + "            "
        for _col in range(size):
            arrow_row += " ↓   "
        print(arrow_row)

        # ─────────────────────────────────────────────────────────────────
        # A SCRATCHPAD + SKEW A (left) + SYSTOLIC ARRAY (center)
        # Data flows: SRAM → In → R(N-1) → ... → R0 → Out → Array
        # So display left-to-right: SRAM, R(max-1), ..., R0, Out→, Array
        # ─────────────────────────────────────────────────────────────────

        # Build header showing the register labels
        # For max_depth=3: "SRAM    R2  R1  R0  Out"
        skew_header = f"{c.BOLD}A SRAM{c.RESET}  "
        for d in range(max_depth - 1, -1, -1):
            skew_header += f" R{d} "
        skew_header += f" Out    │   {c.BOLD}Systolic Array{c.RESET}"
        print(skew_header)

        a_data = [int(x) for x in feeder.sram_a.output_data] if feeder.sram_a.output_valid else None

        # For each row, print: A_SRAM_element | R(N-1) ... R0 | Out → | Array_row
        for row in range(size):
            # A SRAM element for this row
            if a_data is not None:
                a_sram_cell = f"  [{c.YELLOW}{a_data[row]:>2}{c.RESET}]  "
            else:
                a_sram_cell = f"  {c.DIM}[--]{c.RESET}  "

            # Skew A lane: show registers from deepest to shallowest, then output
            depth = row  # Row r has r delay stages
            regs = a_regs[row]

            skew_cells = ""
            # Show registers R(max-1) down to R0
            # FIFO fills from input side: when len < depth, items are at R(depth-len) through R(depth-1)
            # Formula: fifo[i] is at R[depth - len + i], so R[d] corresponds to fifo[d - depth + len]
            fifo_len = len(regs)
            for d in range(max_depth - 1, -1, -1):
                fifo_idx = d - depth + fifo_len  # Map R[d] to FIFO index
                if d < depth and 0 <= fifo_idx < fifo_len:
                    val, valid, sos, eos = regs[fifo_idx]
                    skew_cells += fmt_val(
                        val, valid, sos, eos, c.YELLOW
                    )  # 4 chars to match " R{d} " header
                elif d < depth:
                    skew_cells += f"{c.DIM}  · {c.RESET}"  # 4 chars, dot at position 3
                else:
                    skew_cells += "    "  # 4 spaces for unused positions

            # Output (going to array)
            a_val, a_val_valid = a_outputs[row]
            out_cell = fmt_val(a_val, a_val_valid, a_sos[row], a_eos[row], c.YELLOW)

            # Array row
            array_row = ""
            for col in range(size):
                val = array_c[row, col]
                state = state_grid[row][col]
                done = sa.pes[row][col].done
                array_row += fmt_pe(val, state, done) + " "

            # Combine and print
            arrow = "→" if a_val_valid else " "
            print(f"{a_sram_cell}{skew_cells}{out_cell} {arrow}r{row}│ {array_row}")

        print()

        # Check if all PEs have completed (received EOS)
        if sa.all_done():
            print(f"{c.GREEN}{c.BOLD}✓ All PEs completed!{c.RESET}")
            break

        # Advance to next cycle: step mode waits for Enter, otherwise use delay
        if step:
            print(f"{c.DIM}Press Enter to advance to next cycle...{c.RESET}", end="", flush=True)
            input()
        else:
            time.sleep(delay_ms / 1000.0)

    array_c = sa.get_accumulator()
    print(f"\n{c.BOLD}Final Result:{c.RESET}")
    print(array_c)
    print(f"\n{c.BOLD}Expected:{c.RESET}")
    print(A @ B)

    if np.array_equal(array_c, A @ B):
        print(f"\n{c.GREEN}{c.BOLD}✓ PASS{c.RESET}")
    else:
        print(f"\n{c.RED}{c.BOLD}✗ FAIL{c.RESET}")

    # ═══════════════════════════════════════════════════════════════════════
    # TIMELINE LOG (post-animation summary)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 76}")
    print(f"{c.BOLD}EXECUTION TIMELINE LOG{c.RESET}")
    print(f"{'═' * 76}")
    print(f"Configuration: {size}×{size} array, {sram_latency}-cycle SRAM latency, K={k_dim}")
    print(f"{'─' * 76}")

    for cycle, events in timeline_log:
        print(f"{c.BOLD}Cycle {cycle:2d}{c.RESET} │ ", end="")
        if len(events) == 1:
            print(events[0])
        else:
            print(events[0])
            for event in events[1:]:
                print(f"         │ {event}")

    print(f"{'─' * 76}")
    total_cycles = timeline_log[-1][0] + 1 if timeline_log else 0
    print(f"Total cycles: {total_cycles}")
    print(
        f"Throughput: {size * size} results in {total_cycles} cycles = {size * size / total_cycles:.2f} results/cycle"
    )
    print(f"{'═' * 76}")


def main():
    parser = argparse.ArgumentParser(
        description="Skew Buffer and SRAM Timing Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # GEMM dimensions (systolic array requires M=N for square array)
    add_gemm_args(parser, default_m=4, default_n=4, default_k=4)

    # Skew buffer configuration
    config_group = parser.add_argument_group("Skew Buffer Configuration")
    config_group.add_argument(
        "--latency", type=int, default=4, help="SRAM latency cycles (default: 4)"
    )

    # Display mode selection
    mode_group = parser.add_argument_group("Display Mode")
    mode_group.add_argument("--timing", action="store_true", help="Show timing diagram only")
    mode_group.add_argument("--layout", action="store_true", help="Show memory layout only")
    mode_group.add_argument("--animate", action="store_true", help="Run animation")

    # Common animation arguments (--movie skips prompts for term2svg capture)
    add_animation_args(
        parser,
        include_fast=True,
        include_max_cycles=False,
        include_movie=True,
    )

    args = parser.parse_args()

    # Systolic array requires square dimensions (M=N)
    if args.m != args.n:
        print(f"Error: Systolic array requires M=N (got M={args.m}, N={args.n})")
        sys.exit(1)

    array_size = args.m  # M=N for square array
    k_dim = args.k

    use_color = not args.no_color
    delay = get_effective_delay(args)

    if args.timing:
        draw_timing_diagram(array_size, args.latency, k_dim, use_color)
    elif args.layout:
        draw_memory_layout(array_size, use_color)
    elif args.animate:
        animate_feeding(array_size, args.latency, k_dim, delay, use_color, args.movie, args.step)
    else:
        # Show all by default
        draw_memory_layout(array_size, use_color)
        print("\n" + "=" * 76 + "\n")
        draw_timing_diagram(array_size, args.latency, k_dim, use_color)
        print(
            f"\n{Colors.DIM if use_color else ''}Run with --animate for step-by-step animation{Colors.RESET if use_color else ''}"
        )


if __name__ == "__main__":
    main()
