#!/usr/bin/env python3
"""
Memory Subsystem and Systolic Array Dataflow Visualization.

This example visualizes the complete data flow architecture:
1. Scratchpad memory with multiple banks
2. Row/column streaming interface to the systolic array
3. Result accumulator with output streaming

The visualization shows how data moves through the entire system:
    DRAM → Scratchpad → SystolicArray → Accumulator → DRAM

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                        DRAM (External)                             │
    └───────────────────┬────────────────────────────┬───────────────────┘
                        │ Load                       │ Store
                        ↓                            ↑
    ┌───────────────────────────────┐    ┌───────────────────────────────┐
    │         SCRATCHPAD            │    │        ACCUMULATOR            │
    │  ┌──────┐┌──────┐┌──────┐     │    │  ┌──────┐┌──────┐┌──────┐     │
    │  │Bank 0││Bank 1││Bank 2│...  │    │  │Bank 0││Bank 1││Bank 2│ ... │
    │  └──────┘└──────┘└──────┘     │    │  └──────┘└──────┘└──────┘     │
    └───────────┬───────────────────┘    └─────────────────┬─────────────┘
                │                                          ↑
         Row A  │  Col B                                   │ Results
         Stream ↓  Stream                                  │
            ┌─────────────────────────────────────────┐    │
            │          SYSTOLIC ARRAY                 │    │
     A[i] → │  ┌────┐ ┌────┐ ┌────┐ ┌────┐            │    │
            │  │ PE │→│ PE │→│ PE │→│ PE │ → (drain)  │    │
            │  └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘            │    │
     B[j] ↓ │    ↓       ↓       ↓       ↓            │    │
            │  ┌────┐ ┌────┐ ┌────┐ ┌────┐            │    │
            │  │ PE │→│ PE │→│ PE │→│ PE │            │────┘
            │  └────┘ └────┘ └────┘ └────┘            │
            └─────────────────────────────────────────┘

Usage:
    python 04_memory_dataflow.py [--size N] [--banks N] [--delay MS]
    python 04_memory_dataflow.py --diagram  # Show static architecture diagram

Requirements:
    pip install matplotlib
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

# =============================================================================
# ANSI Color Codes
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# Memory Models
# =============================================================================


class BankState(Enum):
    IDLE = auto()
    READ = auto()
    WRITE = auto()


@dataclass
class ScratchpadBank:
    """Model of a single scratchpad bank."""

    id: int
    capacity: int  # Number of rows
    width: int  # Elements per row (matches array dimension)
    data: np.ndarray = None
    state: BankState = BankState.IDLE
    last_addr: int = 0

    def __post_init__(self):
        if self.data is None:
            self.data = np.zeros((self.capacity, self.width), dtype=np.int8)

    def write_row(self, addr: int, row_data: np.ndarray):
        """Write a row to the bank."""
        self.data[addr, : len(row_data)] = row_data
        self.state = BankState.WRITE
        self.last_addr = addr

    def read_row(self, addr: int) -> np.ndarray:
        """Read a row from the bank."""
        self.state = BankState.READ
        self.last_addr = addr
        return self.data[addr].copy()


@dataclass
class Scratchpad:
    """Multi-bank scratchpad memory."""

    num_banks: int
    bank_capacity: int
    width: int  # Elements per row
    banks: list = field(default_factory=list)

    def __post_init__(self):
        self.banks = [
            ScratchpadBank(id=i, capacity=self.bank_capacity, width=self.width)
            for i in range(self.num_banks)
        ]

    def load_matrix_a(self, A: np.ndarray):
        """Load matrix A into scratchpad (row-wise into bank 0)."""
        for i, row in enumerate(A):
            self.banks[0].write_row(i, row)

    def load_matrix_b(self, B: np.ndarray):
        """Load matrix B into scratchpad (column-wise into bank 1)."""
        for j in range(B.shape[1]):
            self.banks[1].write_row(j, B[:, j])

    def stream_a_row(self, row_idx: int) -> np.ndarray:
        """Stream a row of A from scratchpad."""
        return self.banks[0].read_row(row_idx)

    def stream_b_col(self, col_idx: int) -> np.ndarray:
        """Stream a column of B from scratchpad."""
        return self.banks[1].read_row(col_idx)

    def reset_states(self):
        """Reset all bank states to idle."""
        for bank in self.banks:
            bank.state = BankState.IDLE


@dataclass
class AccumulatorBank:
    """Accumulator bank for results."""

    id: int
    capacity: int
    width: int
    data: np.ndarray = None
    state: BankState = BankState.IDLE
    last_addr: int = 0

    def __post_init__(self):
        if self.data is None:
            self.data = np.zeros((self.capacity, self.width), dtype=np.int32)

    def accumulate(self, addr: int, values: np.ndarray):
        """Accumulate values into a row."""
        self.data[addr, : len(values)] += values
        self.state = BankState.WRITE
        self.last_addr = addr

    def read_row(self, addr: int) -> np.ndarray:
        """Read accumulated results."""
        self.state = BankState.READ
        self.last_addr = addr
        return self.data[addr].copy()


@dataclass
class Accumulator:
    """Multi-bank accumulator memory."""

    num_banks: int
    bank_capacity: int
    width: int
    banks: list = field(default_factory=list)

    def __post_init__(self):
        self.banks = [
            AccumulatorBank(id=i, capacity=self.bank_capacity, width=self.width)
            for i in range(self.num_banks)
        ]

    def store_result_row(self, row_idx: int, values: np.ndarray):
        """Store a row of results."""
        bank_idx = row_idx % self.num_banks
        addr = row_idx // self.num_banks
        self.banks[bank_idx].accumulate(addr, values)

    def get_result_matrix(self, rows: int) -> np.ndarray:
        """Extract full result matrix."""
        result = np.zeros((rows, self.width), dtype=np.int32)
        for i in range(rows):
            bank_idx = i % self.num_banks
            addr = i // self.num_banks
            result[i] = self.banks[bank_idx].data[addr]
        return result

    def reset_states(self):
        for bank in self.banks:
            bank.state = BankState.IDLE


# =============================================================================
# PE and Systolic Array (simplified for visualization)
# =============================================================================


@dataclass
class PE:
    """Processing element state."""

    a: int = 0
    b: int = 0
    c: int = 0
    a_valid: bool = False
    b_valid: bool = False
    active: bool = False


@dataclass
class SystolicArraySim:
    """Simplified systolic array simulator focused on I/O streaming."""

    size: int
    pes: list = field(default_factory=list)
    cycle: int = 0

    # Current input streams
    a_stream: np.ndarray = None  # Current row being fed
    b_stream: np.ndarray = None  # Current column being fed
    a_stream_idx: int = 0
    b_stream_idx: int = 0

    def __post_init__(self):
        self.pes = [[PE() for _ in range(self.size)] for _ in range(self.size)]

    def reset(self):
        """Reset array state."""
        for row in self.pes:
            for pe in row:
                pe.a = pe.b = pe.c = 0
                pe.a_valid = pe.b_valid = pe.active = False
        self.cycle = 0

    def feed_row(self, a_values: np.ndarray):
        """Feed a row of A values to left edge."""
        for i in range(min(len(a_values), self.size)):
            self.pes[i][0].a = a_values[i]
            self.pes[i][0].a_valid = True

    def feed_col(self, b_values: np.ndarray):
        """Feed a column of B values to top edge."""
        for j in range(min(len(b_values), self.size)):
            self.pes[0][j].b = b_values[j]
            self.pes[0][j].b_valid = True

    def step(self):
        """Execute one cycle of systolic operation."""
        n = self.size

        # Shift and compute
        shifted_a = [[None] * n for _ in range(n)]
        shifted_b = [[None] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                pe = self.pes[i][j]

                if pe.a_valid and pe.b_valid:
                    pe.c += pe.a * pe.b
                    pe.active = True
                else:
                    pe.active = False

                if j + 1 < n and pe.a_valid:
                    shifted_a[i][j + 1] = pe.a

                if i + 1 < n and pe.b_valid:
                    shifted_b[i + 1][j] = pe.b

        # Apply shifts
        for i in range(n):
            for j in range(n):
                pe = self.pes[i][j]
                if j > 0:
                    if shifted_a[i][j] is not None:
                        pe.a = shifted_a[i][j]
                        pe.a_valid = True
                    else:
                        pe.a_valid = False

                if i > 0:
                    if shifted_b[i][j] is not None:
                        pe.b = shifted_b[i][j]
                        pe.b_valid = True
                    else:
                        pe.b_valid = False

        # Clear edge inputs (they were consumed)
        for i in range(n):
            if shifted_a[i][0] is None:
                self.pes[i][0].a_valid = False
        for j in range(n):
            if shifted_b[0][j] is None:
                self.pes[0][j].b_valid = False

        self.cycle += 1

    def drain_results(self) -> np.ndarray:
        """Extract accumulated results."""
        result = np.zeros((self.size, self.size), dtype=np.int32)
        for i in range(self.size):
            for j in range(self.size):
                result[i, j] = self.pes[i][j].c
        return result


# =============================================================================
# Visualization Functions
# =============================================================================


def draw_bank(
    bank_id: int,
    state: BankState,
    data_preview: list,
    last_addr: int,
    width: int = 24,
) -> list[str]:
    """Draw a memory bank as ASCII art."""
    c = Colors

    state_str = {
        BankState.IDLE: f"{c.GRAY}IDLE{c.RESET}",
        BankState.READ: f"{c.GREEN}READ{c.RESET}",
        BankState.WRITE: f"{c.YELLOW}WRITE{c.RESET}",
    }[state]

    lines = []
    lines.append(f"┌{'─' * (width - 2)}┐")
    lines.append(f"│{f'Bank {bank_id}':^{width - 2}}│")
    lines.append(f"│{state_str:^{width + 7}}│")
    lines.append(f"├{'─' * (width - 2)}┤")

    # Show a few data rows
    for i, row in enumerate(data_preview[:4]):
        marker = "→" if i == last_addr and state != BankState.IDLE else " "
        row_str = ",".join(f"{v:2d}" for v in row[:3])
        if len(row) > 3:
            row_str += "..."
        display = f"{marker}{row_str}"
        lines.append(f"│{display:<{width - 2}}│")

    if len(data_preview) > 4:
        lines.append(f"│{'...':^{width - 2}}│")

    lines.append(f"└{'─' * (width - 2)}┘")
    return lines


def draw_systolic_array(array: SystolicArraySim, compact: bool = False) -> list[str]:
    """Draw the systolic array state."""
    c = Colors
    n = array.size
    lines = []

    cell_width = 6 if compact else 8

    # Header with B input indicators
    header = "      "
    for j in range(n):
        pe = array.pes[0][j]
        if pe.b_valid:
            header += f"{c.CYAN}B[{j}]↓{c.RESET}" + " " * (cell_width - 4)
        else:
            header += " " * cell_width
    lines.append(header)

    # Array rows
    for i in range(n):
        # A input indicator
        pe = array.pes[i][0]
        row_prefix = f"{c.YELLOW}A[{i}]→{c.RESET}" if pe.a_valid else "     "

        row = row_prefix + " "

        for j in range(n):
            pe = array.pes[i][j]

            if pe.active:
                bg = c.BG_GREEN
                content = f"{pe.a}×{pe.b}"
            elif pe.a_valid and pe.b_valid:
                bg = c.BG_CYAN
                content = f"{pe.a},{pe.b}"
            elif pe.a_valid:
                bg = c.BG_YELLOW
                content = f"a={pe.a}"
            elif pe.b_valid:
                bg = c.BG_BLUE
                content = f"b={pe.b}"
            else:
                bg = ""
                content = "·"

            cell = f"{bg}[{content:^{cell_width - 2}}]{c.RESET}"
            row += cell

        lines.append(row)

    # Accumulator values
    lines.append("")
    lines.append(f"  {c.GREEN}Accumulated C:{c.RESET}")
    for i in range(n):
        row = "      "
        for j in range(n):
            pe = array.pes[i][j]
            val = pe.c
            if val > 0:
                row += f"{c.GREEN}{val:^{cell_width}}{c.RESET}"
            else:
                row += f"{val:^{cell_width}}"
        lines.append(row)

    return lines


def draw_architecture_diagram() -> str:
    """Draw the full system architecture diagram."""
    c = Colors

    diagram = f"""
{c.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SYSTARS MEMORY DATAFLOW ARCHITECTURE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝{c.RESET}

{c.GRAY}
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DRAM (External Memory)                          │
│                        Matrices A, B stored row-major                        │
└───────────────────────────┬──────────────────────────┬───────────────────────┘{c.RESET}
                            │                          │
                   {c.CYAN}Load Controller{c.RESET}           {c.MAGENTA}Store Controller{c.RESET}
                         (DMA)                       (DMA)
                            │                          ↑
                            ↓                          │
{c.YELLOW}┌─────────────────────────────────────────┐{c.RESET}  {c.GREEN}┌─────────────────────────────────────────┐{c.RESET}
{c.YELLOW}│            SCRATCHPAD MEMORY            │{c.RESET}  {c.GREEN}│            ACCUMULATOR MEMORY           │{c.RESET}
{c.YELLOW}│                                         │{c.RESET}  {c.GREEN}│                                         │{c.RESET}
{c.YELLOW}│  ┌────────┐ ┌────────┐ ┌────────┐       │{c.RESET}  {c.GREEN}│  ┌────────┐ ┌────────┐ ┌────────┐       │{c.RESET}
{c.YELLOW}│  │ Bank 0 │ │ Bank 1 │ │ Bank 2 │  ...  │{c.RESET}  {c.GREEN}│  │ Bank 0 │ │ Bank 1 │ │ Bank 2 │  ...  │{c.RESET}
{c.YELLOW}│  │ (A)    │ │ (B)    │ │        │       │{c.RESET}  {c.GREEN}│  │ (C)    │ │        │ │        │       │{c.RESET}
{c.YELLOW}│  └────┬───┘ └────┬───┘ └────────┘       │{c.RESET}  {c.GREEN}│  └────────┘ └────────┘ └────────┘       │{c.RESET}
{c.YELLOW}│       │          │                      │{c.RESET}  {c.GREEN}│       ↑                                 │{c.RESET}
{c.YELLOW}└───────┼──────────┼──────────────────────┘{c.RESET}  {c.GREEN}└───────┼─────────────────────────────────┘{c.RESET}
        │          │                                 │
 {c.YELLOW}A row{c.RESET}  │    {c.CYAN}B col{c.RESET} │                           {c.GREEN}C row{c.RESET} │
 {c.YELLOW}stream{c.RESET} │   {c.CYAN}stream{c.RESET} │                          {c.GREEN}stream{c.RESET} │
        ↓          ↓                                 │
      ┌──────────────────────────────────────────┐   │
      │                {c.BOLD}SYSTOLIC ARRAY{c.RESET}            │   │
      │                                          │   │
      │   {c.CYAN}B[0]{c.RESET}      {c.CYAN}B[1]{c.RESET}      {c.CYAN}B[2]{c.RESET}      {c.CYAN}B[3]{c.RESET}     │   │
      │     ↓         ↓         ↓         ↓      │   │
      │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │   │
{c.YELLOW}A[0]→{c.RESET} │  │ PE00 │─→│ PE01 │─→│ PE02 │─→│ PE03 │──┼───────→ {c.DIM}drain{c.RESET}
      │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  │   │
      │     │         │         │         │      │   │
      │  ┌──↓───┐  ┌──↓───┐  ┌──↓───┐  ┌──↓───┐  │   │
{c.YELLOW}A[1]→{c.RESET} │  │ PE10 │─→│ PE11 │─→│ PE12 │─→│ PE13 │  │───↑ {c.GREEN}C[row]{c.RESET}
      │  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  │
      │     │         │         │         │      │
      │  ┌──↓───┐  ┌──↓───┐  ┌──↓───┐  ┌──↓───┐  │
{c.YELLOW}A[2]→{c.RESET} │  │ PE20 │─→│ PE21 │─→│ PE22 │─→│ PE23 │  │
      │  └──────┘  └──────┘  └──────┘  └──────┘  │
      └──────────────────────────────────────────┘

{c.BOLD}Data Flow Sequence:{c.RESET}
  1. {c.CYAN}LOAD{c.RESET}:    DRAM → Scratchpad (via StreamReader DMA)
  2. {c.YELLOW}COMPUTE{c.RESET}: Scratchpad → Systolic Array → Accumulator
     • A rows stream from left edge (one element per row per cycle)
     • B cols stream from top edge (one element per column per cycle)
     • Results accumulate in PEs, then drain to accumulator
  3. {c.MAGENTA}STORE{c.RESET}:   Accumulator → DRAM (via StreamWriter DMA)

{c.BOLD}PE Operation:{c.RESET}  c += a x b  (Multiply-Accumulate)
  - In Output-Stationary mode: a,b flow through; c accumulates in place
  - In Weight-Stationary mode: b stays; a,c flow through

{c.BOLD}Scratchpad Organization:{c.RESET}
  - Multiple banks for parallel access (typically 4)
  - Each row stores one vector (row of A or column of B)
  - Width = array_size x element_bits (e.g., 16x8 = 128 bits)

{c.BOLD}Address Mapping:{c.RESET}
  - Bit 31: is_accumulator (0=scratchpad, 1=accumulator)
  - Bit 30: accumulate mode (for writes)
  - Bits [bank_bits-1:0]: bank selection
  - Remaining bits: row address within bank
"""
    return diagram


def visualize_dataflow(
    size: int = 4,
    num_banks: int = 4,
    delay_ms: int = 500,
    use_color: bool = True,
):
    """Run animated dataflow visualization."""
    if not use_color:
        Colors.disable()

    c = Colors

    # Create test matrices
    np.random.seed(42)
    A = np.random.randint(1, 5, size=(size, size), dtype=np.int8)
    B = np.random.randint(1, 5, size=(size, size), dtype=np.int8)
    C_expected = A.astype(np.int32) @ B.astype(np.int32)

    # Initialize memory subsystems
    scratchpad = Scratchpad(num_banks=num_banks, bank_capacity=64, width=size)
    array = SystolicArraySim(size=size)

    # Phase 1: Load matrices into scratchpad
    clear_screen()
    print(f"{c.BOLD}{'═' * 70}{c.RESET}")
    print(f"{c.BOLD}Phase 1: LOAD - DMA Transfer DRAM → Scratchpad{c.RESET}")
    print(f"{'═' * 70}\n")

    print("Matrix A (stored in Bank 0, row-major):")
    print(A)
    print("\nMatrix B (stored in Bank 1, column-major):")
    print(B)

    scratchpad.load_matrix_a(A)
    scratchpad.load_matrix_b(B)

    print(f"\n{c.GREEN}✓ Matrices loaded into scratchpad{c.RESET}")
    print(f"\n{c.DIM}Press Enter to continue to compute phase...{c.RESET}")
    input()

    # Phase 2: Compute with streaming visualization
    print(f"\n{c.BOLD}{'═' * 70}{c.RESET}")
    print(f"{c.BOLD}Phase 2: COMPUTE - Systolic Array Execution{c.RESET}")
    print(f"{'═' * 70}\n")

    # Create skewed input queues (same as wavefront demo)
    a_queue = []
    for i in range(size):
        row_data = [None] * i + list(A[i, :]) + [None] * (size - 1)
        a_queue.append(row_data)

    b_queue = []
    for j in range(size):
        col_data = [None] * j + list(B[:, j]) + [None] * (size - 1)
        b_queue.append(col_data)

    total_cycles = 2 * size - 1 + size

    for cycle in range(total_cycles):
        clear_screen()
        print(f"{c.BOLD}{'═' * 70}{c.RESET}")
        print(f"{c.BOLD}Phase 2: COMPUTE - Cycle {cycle}{c.RESET}")
        print(f"{'═' * 70}\n")

        # Show scratchpad state
        print(f"{c.YELLOW}SCRATCHPAD:{c.RESET}")
        scratchpad.reset_states()

        # Determine what's being read this cycle
        for i in range(size):
            if cycle < len(a_queue[i]) and a_queue[i][cycle] is not None:
                scratchpad.banks[0].state = BankState.READ
                scratchpad.banks[0].last_addr = i
                break

        for j in range(size):
            if cycle < len(b_queue[j]) and b_queue[j][cycle] is not None:
                scratchpad.banks[1].state = BankState.READ
                scratchpad.banks[1].last_addr = j
                break

        # Draw banks
        bank0_lines = draw_bank(
            0,
            scratchpad.banks[0].state,
            A.tolist(),
            scratchpad.banks[0].last_addr,
        )
        bank1_lines = draw_bank(
            1,
            scratchpad.banks[1].state,
            B.T.tolist(),  # Transposed for column storage
            scratchpad.banks[1].last_addr,
        )

        max_lines = max(len(bank0_lines), len(bank1_lines))
        for i in range(max_lines):
            line0 = bank0_lines[i] if i < len(bank0_lines) else " " * 14
            line1 = bank1_lines[i] if i < len(bank1_lines) else ""
            print(f"  {line0}    {line1}")

        # Show streaming indicators
        print(f"\n{c.BOLD}Streaming to Array:{c.RESET}")
        a_feed = []
        b_feed = []
        for i in range(size):
            if cycle < len(a_queue[i]) and a_queue[i][cycle] is not None:
                a_feed.append(f"A[{i}]={a_queue[i][cycle]}")
        for j in range(size):
            if cycle < len(b_queue[j]) and b_queue[j][cycle] is not None:
                b_feed.append(f"B[{j}]={b_queue[j][cycle]}")

        if a_feed:
            print(f"  {c.YELLOW}Row stream →{c.RESET} {', '.join(a_feed)}")
        if b_feed:
            print(f"  {c.CYAN}Col stream ↓{c.RESET} {', '.join(b_feed)}")

        # Feed data to array
        for i in range(size):
            if cycle < len(a_queue[i]) and a_queue[i][cycle] is not None:
                array.pes[i][0].a = a_queue[i][cycle]
                array.pes[i][0].a_valid = True
            else:
                array.pes[i][0].a_valid = False

        for j in range(size):
            if cycle < len(b_queue[j]) and b_queue[j][cycle] is not None:
                array.pes[0][j].b = b_queue[j][cycle]
                array.pes[0][j].b_valid = True
            else:
                array.pes[0][j].b_valid = False

        # Show array state
        print(f"\n{c.BOLD}SYSTOLIC ARRAY:{c.RESET}")
        for line in draw_systolic_array(array, compact=True):
            print(f"  {line}")

        array.step()

        time.sleep(delay_ms / 1000.0)

    # Final state
    clear_screen()
    print(f"{c.BOLD}{'═' * 70}{c.RESET}")
    print(f"{c.BOLD}Phase 2: COMPUTE - Complete{c.RESET}")
    print(f"{'═' * 70}\n")

    result = array.drain_results()

    print(f"{c.GREEN}ACCUMULATOR - Final Results:{c.RESET}")
    print(result)

    print(f"\n{c.BOLD}Expected C = A @ B:{c.RESET}")
    print(C_expected)

    if np.array_equal(result, C_expected):
        print(f"\n{c.GREEN}{c.BOLD}✓ PASS: Results match!{c.RESET}")
    else:
        print(f"\n{c.RED}{c.BOLD}✗ FAIL: Results differ!{c.RESET}")

    print(f"\n{c.DIM}Press Enter to see Phase 3 (Store)...{c.RESET}")
    input()

    # Phase 3: Store results
    clear_screen()
    print(f"{c.BOLD}{'═' * 70}{c.RESET}")
    print(f"{c.BOLD}Phase 3: STORE - DMA Transfer Accumulator → DRAM{c.RESET}")
    print(f"{'═' * 70}\n")

    print("Result rows streamed from accumulator to external DRAM:")
    for i in range(size):
        print(f"  {c.GREEN}C[{i}] →{c.RESET} {result[i].tolist()}")
        time.sleep(200 / 1000.0)

    print(f"\n{c.GREEN}{c.BOLD}✓ Complete!{c.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory Subsystem and Dataflow Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=4, help="Array size (default: 4)")
    parser.add_argument("--banks", type=int, default=4, help="Number of memory banks (default: 4)")
    parser.add_argument(
        "--delay", type=int, default=750, help="Delay between frames in ms (default: 750)"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--diagram", action="store_true", help="Show static architecture diagram and exit"
    )

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    if args.diagram:
        print(draw_architecture_diagram())
        return

    visualize_dataflow(
        size=args.size,
        num_banks=args.banks,
        delay_ms=args.delay,
        use_color=not args.no_color,
    )


if __name__ == "__main__":
    main()
