#!/usr/bin/env python3
"""
Transpose DMA Visualization.

Demonstrates how matrix A is transposed during DMA load from DRAM to scratchpad.

For systolic array matmul C = A @ B:
- A (M×K) must be stored column-major in scratchpad (transposed from DRAM)
- B (K×N) stored row-major in scratchpad (direct copy from DRAM)

The transpose DMA uses strided memory access:
- To read column c of A: base_addr + c, stride = K (row width)
- Each gathered column becomes a contiguous row in scratchpad

Usage:
    python 06_transpose_dma.py [--m M] [--k K] [--n N]
    python 06_transpose_dma.py --animate  # Step-by-step animation
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np


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

    @classmethod
    def disable(cls):
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# DMA Models
# =============================================================================


@dataclass
class DRAM:
    """Simulated DRAM with row-major matrix storage."""

    data: dict = field(default_factory=dict)

    def store_matrix(self, _name: str, matrix: np.ndarray, base_addr: int):
        """Store a matrix in row-major order."""
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                addr = base_addr + i * cols + j
                self.data[addr] = matrix[i, j]
        return base_addr, rows, cols

    def read(self, addr: int) -> int:
        """Read a single element."""
        return self.data.get(addr, 0)

    def read_contiguous(self, base_addr: int, count: int) -> list:
        """Read contiguous elements (for row access)."""
        return [self.data.get(base_addr + i, 0) for i in range(count)]

    def read_strided(self, base_addr: int, count: int, stride: int) -> list:
        """Read strided elements (for column access)."""
        return [self.data.get(base_addr + i * stride, 0) for i in range(count)]


@dataclass
class Scratchpad:
    """Simulated scratchpad memory."""

    banks: dict = field(default_factory=dict)

    def write_row(self, bank: int, addr: int, data: list):
        """Write a row to a bank."""
        if bank not in self.banks:
            self.banks[bank] = {}
        self.banks[bank][addr] = data.copy()

    def read_row(self, bank: int, addr: int) -> list:
        """Read a row from a bank."""
        if bank in self.banks and addr in self.banks[bank]:
            return self.banks[bank][addr]
        return []

    def get_matrix(self, bank: int, num_rows: int) -> np.ndarray:
        """Reconstruct matrix from bank."""
        if bank not in self.banks:
            return np.array([])
        rows = []
        for i in range(num_rows):
            if i in self.banks[bank]:
                rows.append(self.banks[bank][i])
        return np.array(rows, dtype=np.int32)


@dataclass
class TransposeDMA:
    """DMA engine with transpose support."""

    dram: DRAM
    scratchpad: Scratchpad
    sram_latency: int = 4

    # Transfer state
    transfers: list = field(default_factory=list)

    def load_contiguous(
        self, dram_addr: int, sp_bank: int, sp_addr: int, rows: int, cols: int
    ) -> list:
        """
        Load matrix contiguously (no transpose).

        Reads row-by-row from DRAM, writes row-by-row to scratchpad.
        """
        transfers = []
        for row in range(rows):
            src_addr = dram_addr + row * cols
            data = self.dram.read_contiguous(src_addr, cols)
            self.scratchpad.write_row(sp_bank, sp_addr + row, data)
            transfers.append(
                {
                    "type": "contiguous",
                    "src_row": row,
                    "src_addr": src_addr,
                    "dst_addr": sp_addr + row,
                    "data": data,
                }
            )
        return transfers

    def load_transpose(
        self, dram_addr: int, sp_bank: int, sp_addr: int, src_rows: int, src_cols: int
    ) -> list:
        """
        Load matrix with transpose.

        Reads column-by-column from DRAM (strided), writes as rows to scratchpad.
        """
        transfers = []
        for col in range(src_cols):
            # Strided read: gather column 'col'
            src_addr = dram_addr + col  # Start at element A[0, col]
            stride = src_cols  # Skip to next row
            data = self.dram.read_strided(src_addr, src_rows, stride)
            # Write as row 'col' in scratchpad
            self.scratchpad.write_row(sp_bank, sp_addr + col, data)
            transfers.append(
                {
                    "type": "transpose",
                    "src_col": col,
                    "src_addr": src_addr,
                    "stride": stride,
                    "dst_addr": sp_addr + col,
                    "data": data,
                    "addresses_read": [src_addr + i * stride for i in range(src_rows)],
                }
            )
        return transfers


# =============================================================================
# Visualization
# =============================================================================


def visualize_transpose(
    m: int = 4,
    k: int = 3,
    n: int = 4,
    use_color: bool = True,
):
    """Visualize the transpose DMA operation."""
    if not use_color:
        Colors.disable()
    c = Colors

    print(f"\n{c.BOLD}╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                    TRANSPOSE DMA VISUALIZATION                           ║")
    print(
        f"╚══════════════════════════════════════════════════════════════════════════╝{c.RESET}\n"
    )

    # Create test matrices
    np.random.seed(42)
    A = np.arange(1, m * k + 1, dtype=np.int8).reshape(m, k)
    B = np.arange(1, k * n + 1, dtype=np.int8).reshape(k, n) * 10

    print(f"{c.BOLD}Goal:{c.RESET} Prepare matrices for systolic matmul C = A @ B")
    print(f"  A: {m}×{k} matrix (M×K)")
    print(f"  B: {k}×{n} matrix (K×N)")
    print()

    # Show DRAM layout
    print(f"{c.BOLD}Step 1: DRAM Layout (Row-Major){c.RESET}")
    print()

    print(f"  {c.YELLOW}Matrix A in DRAM:{c.RESET}")
    for i in range(m):
        row_str = f"    Row {i}: "
        for j in range(k):
            addr = i * k + j
            row_str += f"[{addr:2d}]={A[i, j]:2d}  "
        print(row_str)
    print()

    print(f"  {c.CYAN}Matrix B in DRAM:{c.RESET}")
    b_base = 100  # Arbitrary base address for B
    for i in range(k):
        row_str = f"    Row {i}: "
        for j in range(n):
            addr = b_base + i * n + j
            row_str += f"[{addr:2d}]={B[i, j]:3d}  "
        print(row_str)
    print()

    # Initialize DMA system
    dram = DRAM()
    scratchpad = Scratchpad()
    dma = TransposeDMA(dram=dram, scratchpad=scratchpad)

    # Store matrices in DRAM
    dram.store_matrix("A", A, 0)
    dram.store_matrix("B", B, b_base)

    # Show transpose operation
    print(f"{c.BOLD}Step 2: DMA Transfer with Transpose for A{c.RESET}")
    print()
    print(f"  {c.YELLOW}Reading columns of A with strided access:{c.RESET}")
    print()

    for col in range(k):
        src_addr = col
        stride = k
        addresses = [src_addr + i * stride for i in range(m)]
        values = [A[i, col] for i in range(m)]

        print(f"    Column {col}:")
        print(f"      Addresses: {addresses}")
        print(f"      Values:    {values}")
        print(f"      → Scratchpad row {col}: {values}")
        print()

    # Perform transpose
    dma.load_transpose(0, 0, 0, m, k)

    # Show scratchpad layout after A transpose
    print(f"{c.BOLD}Step 3: Scratchpad Layout After Transpose{c.RESET}")
    print()

    print(f"  {c.YELLOW}Scratchpad Bank 0 (A, column-major):{c.RESET}")
    for row in range(k):
        data = scratchpad.read_row(0, row)
        print(f"    Addr {row}: {data}  ← was column {row} of A")
    print()

    # Load B directly
    dma.load_contiguous(b_base, 1, 0, k, n)

    print(f"  {c.CYAN}Scratchpad Bank 1 (B, row-major):{c.RESET}")
    for row in range(k):
        data = scratchpad.read_row(1, row)
        print(f"    Addr {row}: {data}  ← row {row} of B")
    print()

    # Verify for systolic feeding
    print(f"{c.BOLD}Step 4: Verification for Systolic Feeding{c.RESET}")
    print()
    print("  For matmul iteration k, the systolic array needs:")
    print("    - Column k of A → feeds to array rows")
    print("    - Row k of B → feeds to array columns")
    print()

    print("  With the transposed layout:")
    for iter_k in range(min(3, k)):
        a_row = scratchpad.read_row(0, iter_k)
        b_row = scratchpad.read_row(1, iter_k)
        print(f"    Iteration k={iter_k}:")
        print(f"      Read SP_A[{iter_k}] = {a_row} (was A column {iter_k})")
        print(f"      Read SP_B[{iter_k}] = {b_row} (B row {iter_k})")
        print()

    # Compute expected result
    C_expected = A @ B

    print(f"{c.BOLD}Step 5: Expected Result C = A @ B{c.RESET}")
    print()
    print(f"  {c.GREEN}C ({m}×{n}):{c.RESET}")
    print(C_expected)
    print()


def animate_transpose(
    m: int = 4,
    k: int = 3,
    delay_ms: int = 500,
    use_color: bool = True,
):
    """Animate the transpose DMA operation step by step."""
    if not use_color:
        Colors.disable()
    c = Colors

    np.random.seed(42)
    A = np.arange(1, m * k + 1, dtype=np.int8).reshape(m, k)

    # Initialize
    dram = DRAM()
    scratchpad = Scratchpad()
    dram.store_matrix("A", A, 0)

    print(f"\n{c.BOLD}Animating Transpose DMA for {m}×{k} matrix{c.RESET}")
    print(f"\n{c.DIM}Press Enter to start...{c.RESET}")
    input()

    for col in range(k):
        clear_screen()

        print(f"{c.BOLD}╔════════════════════════════════════════════════════╗")
        print(f"║  TRANSPOSE DMA - Reading Column {col}                   ║")
        print(f"╚════════════════════════════════════════════════════╝{c.RESET}\n")

        # Show DRAM with current column highlighted
        print(f"{c.BOLD}DRAM (source):{c.RESET}")
        for i in range(m):
            row_str = "  "
            for j in range(k):
                val = A[i, j]
                if j == col:
                    row_str += f"{c.BG_YELLOW}{val:3d}{c.RESET} "
                else:
                    row_str += f"{val:3d} "
            print(row_str)
        print()

        # Show strided read pattern
        print(f"{c.BOLD}Strided Read Pattern:{c.RESET}")
        src_addr = col
        stride = k
        print(f"  Base address: {src_addr}")
        print(f"  Stride: {stride}")
        print("  Addresses: ", end="")
        addresses = []
        for i in range(m):
            addr = src_addr + i * stride
            addresses.append(addr)
            print(f"{c.YELLOW}{addr}{c.RESET}", end=" ")
        print()

        # Gather values
        values = [A[i, col] for i in range(m)]
        print("  Values: ", end="")
        for v in values:
            print(f"{c.GREEN}{v}{c.RESET}", end=" ")
        print()

        # Write to scratchpad
        scratchpad.write_row(0, col, values)

        print(f"\n{c.BOLD}Scratchpad (destination):{c.RESET}")
        for row in range(k):
            data = scratchpad.read_row(0, row)
            if row == col:
                print(f"  {c.BG_GREEN}Addr {row}: {data}{c.RESET} ← just written")
            elif data:
                print(f"  Addr {row}: {data}")
            else:
                print(f"  {c.DIM}Addr {row}: (empty){c.RESET}")

        print()
        time.sleep(delay_ms / 1000.0)

    # Final state
    clear_screen()
    print(f"{c.BOLD}╔════════════════════════════════════════════════════╗")
    print("║  TRANSPOSE DMA - Complete                          ║")
    print(f"╚════════════════════════════════════════════════════╝{c.RESET}\n")

    print(f"{c.BOLD}Original A (row-major):{c.RESET}")
    print(A)

    print(f"\n{c.BOLD}Transposed A in Scratchpad (column-major):{c.RESET}")
    A_sp = scratchpad.get_matrix(0, k)
    print(A_sp)

    # Verify: A_sp should equal A.T
    print(f"\n{c.BOLD}Verification:{c.RESET}")
    expected = A.T
    if np.array_equal(A_sp, expected):
        print(f"  {c.GREEN}✓ Scratchpad contains A transposed correctly{c.RESET}")
    else:
        print(f"  {c.RED}✗ Mismatch!{c.RESET}")
        print(f"  Expected A.T:\n{expected}")


def show_architecture_diagram(use_color: bool = True):
    """Show the complete data flow architecture with transpose."""
    if not use_color:
        Colors.disable()
    c = Colors

    diagram = f"""
{c.BOLD}╔══════════════════════════════════════════════════════════════════════════════╗
║                COMPLETE MATMUL DATA FLOW WITH TRANSPOSE DMA                   ║
╚══════════════════════════════════════════════════════════════════════════════╝{c.RESET}

{c.DIM}DRAM (External Memory){c.RESET}
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Matrix A (M×K, row-major)           Matrix B (K×N, row-major)                  │
│  ┌─────────────────────┐             ┌─────────────────────┐                    │
│  │ A[0,0] A[0,1] A[0,2]│             │ B[0,0] B[0,1] B[0,2]│                    │
│  │ A[1,0] A[1,1] A[1,2]│             │ B[1,0] B[1,1] B[1,2]│                    │
│  │ A[2,0] A[2,1] A[2,2]│             │ B[2,0] B[2,1] B[2,2]│                    │
│  │ A[3,0] A[3,1] A[3,2]│             └─────────────────────┘                    │
│  └─────────────────────┘                                                        │
└─────────────┬───────────────────────────────────┬───────────────────────────────┘
              │                                   │
              │ {c.YELLOW}Strided DMA{c.RESET}                        │ {c.CYAN}Contiguous DMA{c.RESET}
              │ (transpose)                       │ (direct copy)
              │                                   │
              │  Read col 0: addr 0,3,6,9        │  Read row 0: addr 0,1,2
              │  Read col 1: addr 1,4,7,10       │  Read row 1: addr 3,4,5
              │  Read col 2: addr 2,5,8,11       │  Read row 2: addr 6,7,8
              ↓                                   ↓
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SCRATCHPAD MEMORY                                  │
│                                                                                 │
│  {c.YELLOW}Bank 0: A (column-major/transposed){c.RESET}    {c.CYAN}Bank 1: B (row-major){c.RESET}                │
│  ┌────────────────────────────┐         ┌────────────────────────────┐         │
│  │ Addr 0: A[0,0] A[1,0] A[2,0] A[3,0]│  │ Addr 0: B[0,0] B[0,1] B[0,2]│         │
│  │ Addr 1: A[0,1] A[1,1] A[2,1] A[3,1]│  │ Addr 1: B[1,0] B[1,1] B[1,2]│         │
│  │ Addr 2: A[0,2] A[1,2] A[2,2] A[3,2]│  │ Addr 2: B[2,0] B[2,1] B[2,2]│         │
│  └────────────────────────────┘         └────────────────────────────┘         │
│           │                                      │                              │
│           │  {c.YELLOW}For iter k:{c.RESET}                          │                              │
│           │  Read addr k                        │  Read addr k                 │
│           │  → Column k of A                    │  → Row k of B                │
│           ↓                                      ↓                              │
│     ┌───────────────────────────────────────────────────────────────┐          │
│     │                      SKEW BUFFERS                             │          │
│     │  A skew: lane i gets i cycles delay                          │          │
│     │  B skew: lane j gets j cycles delay                          │          │
│     └───────────────────────────────────────────────────────────────┘          │
│                                  │                                              │
└──────────────────────────────────┼──────────────────────────────────────────────┘
                                   ↓
                    ┌─────────────────────────────────┐
                    │       SYSTOLIC ARRAY            │
                    │                                 │
             A[0,k]→│ PE  →  PE  →  PE  →  PE        │
             A[1,k]→│  ↓      ↓      ↓      ↓        │
             A[2,k]→│ PE  →  PE  →  PE  →  PE        │
             A[3,k]→│  ↓      ↓      ↓      ↓        │
                    │ PE  →  PE  →  PE  →  PE        │
                    │ ↑      ↑      ↑      ↑         │
                    │B[k,0] B[k,1] B[k,2] B[k,3]     │
                    └─────────────────────────────────┘

{c.BOLD}Key Insight:{c.RESET}
  - Transpose happens at DMA time, not compute time
  - One strided read gathers entire column → written as row
  - At compute time, single-cycle row reads feed the array
  - This trades DMA bandwidth for compute efficiency
"""
    print(diagram)


def main():
    parser = argparse.ArgumentParser(
        description="Transpose DMA Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--m", type=int, default=4, help="M dimension (default: 4)")
    parser.add_argument("--k", type=int, default=3, help="K dimension (default: 3)")
    parser.add_argument("--n", type=int, default=4, help="N dimension (default: 4)")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("--animate", action="store_true", help="Animate transpose step by step")
    parser.add_argument("--diagram", action="store_true", help="Show architecture diagram")
    parser.add_argument("--delay", type=int, default=500, help="Animation delay ms")

    args = parser.parse_args()
    use_color = not args.no_color

    if args.diagram:
        show_architecture_diagram(use_color)
    elif args.animate:
        animate_transpose(args.m, args.k, args.delay, use_color)
    else:
        visualize_transpose(args.m, args.k, args.n, use_color)


if __name__ == "__main__":
    main()
