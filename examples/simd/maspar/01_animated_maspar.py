#!/usr/bin/env python3
"""
Animated MasPar SIMD GEMM Visualization.

Visualizes matrix multiplication on the MasPar SIMD array processor using
Cannon's algorithm. Shows:
- The 2D PE array with register values
- XNET data shifts (A west, B north)
- Accumulation progress in each PE

The MasPar was a classic SIMD machine from the late 1980s where all PEs
execute the same instruction in lockstep - no divergence handling like
modern GPUs.

Usage:
    python 01_animated_maspar.py              # 4x4 GEMM with animation
    python 01_animated_maspar.py --m 8 --n 8  # 8x8 GEMM
    python 01_animated_maspar.py --fast       # No animation delay
    python 01_animated_maspar.py --step       # Step through manually
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add examples path for common CLI utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.cli import add_animation_args, add_gemm_args, get_effective_delay, should_print

from systars.simd.maspar import (
    Instruction,
    MasParConfig,
    MasParSim,
    Opcode,
)


class MasParAnimator:
    """Animated visualization of MasPar GEMM execution."""

    def __init__(
        self,
        m: int = 4,
        n: int = 4,
        k: int = 4,
        delay_ms: int = 500,
        step_mode: bool = False,
        no_color: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.delay_ms = delay_ms
        self.step_mode = step_mode
        self.no_color = no_color

        # Create simulator with matching array size
        self.config = MasParConfig(array_rows=m, array_cols=n)
        self.sim = MasParSim(self.config)

        # Generate random matrices
        np.random.seed(42)
        self.A = np.random.randint(1, 10, (m, k), dtype=np.int32)
        self.B = np.random.randint(1, 10, (k, n), dtype=np.int32)
        self.expected = self.A @ self.B

        # ANSI colors
        self.RESET = "" if no_color else "\033[0m"
        self.BOLD = "" if no_color else "\033[1m"
        self.DIM = "" if no_color else "\033[2m"
        self.RED = "" if no_color else "\033[31m"
        self.GREEN = "" if no_color else "\033[32m"
        self.YELLOW = "" if no_color else "\033[33m"
        self.BLUE = "" if no_color else "\033[34m"
        self.MAGENTA = "" if no_color else "\033[35m"
        self.CYAN = "" if no_color else "\033[36m"

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def print_header(self, cycle: int, instruction: Instruction | None, iteration: int) -> None:
        """Print animation header with cycle info."""
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print(
            f"{self.BOLD}MasPar SIMD Array - {self.m}x{self.n} GEMM (Cannon's Algorithm){self.RESET}"
        )
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print()
        print(
            f"Cycle: {self.CYAN}{cycle:4d}{self.RESET}  "
            f"Iteration: {self.YELLOW}{iteration + 1}/{self.k}{self.RESET}"
        )
        if instruction:
            print(f"Instruction: {self.GREEN}{instruction}{self.RESET}")
        else:
            print(f"Instruction: {self.DIM}(stalled - multi-cycle op){self.RESET}")

    def print_pe_array(self, highlight_reg: str = "C") -> None:
        """
        Print the PE array state.

        Args:
            highlight_reg: Which register to highlight (A, B, or C)
        """
        reg_idx = {"A": 0, "B": 1, "C": 2}[highlight_reg]

        print(f"{self.BOLD}PE Array - Register {highlight_reg}:{self.RESET}")
        print()

        # Column headers
        header = "     "
        for c in range(self.n):
            header += f"  PE{c:02d} "
        print(f"{self.DIM}{header}{self.RESET}")
        print(f"{self.DIM}     {'-' * (7 * self.n)}{self.RESET}")

        # PE values
        for r in range(self.m):
            row_str = f"{self.DIM}R{r:02d}:{self.RESET} "
            for c in range(self.n):
                pe = self.sim.pe_array.get_pe(r, c)
                val = pe.get_register(reg_idx)
                if highlight_reg == "C":
                    # Color C values based on expected match
                    expected_val = self.expected[r, c]
                    if val == expected_val:
                        row_str += f"{self.GREEN}{val:6d}{self.RESET} "
                    else:
                        row_str += f"{self.YELLOW}{val:6d}{self.RESET} "
                elif highlight_reg == "A":
                    row_str += f"{self.BLUE}{val:6d}{self.RESET} "
                else:  # B
                    row_str += f"{self.MAGENTA}{val:6d}{self.RESET} "
            print(row_str)
        print()

    def print_matrices(self) -> None:
        """Print input matrices and expected result."""
        print(f"{self.BOLD}Input Matrices:{self.RESET}")
        print(f"\n{self.BLUE}A ({self.m}x{self.k}):{self.RESET}")
        for row in self.A:
            print("  " + " ".join(f"{v:3d}" for v in row))

        print(f"\n{self.MAGENTA}B ({self.k}x{self.n}):{self.RESET}")
        for row in self.B:
            print("  " + " ".join(f"{v:3d}" for v in row))

        print(f"\n{self.GREEN}Expected C = A @ B ({self.m}x{self.n}):{self.RESET}")
        for row in self.expected:
            print("  " + " ".join(f"{v:4d}" for v in row))
        print()

    def print_xnet_status(self, instruction: Instruction | None) -> None:
        """Print XNET status line (always prints 3 lines to maintain consistent layout)."""
        print()  # Blank line after instruction
        if instruction and instruction.opcode == Opcode.XNET_E:
            print(f"{self.CYAN}XNET: A shifts WEST  <-- <-- <-- (toroidal){self.RESET}")
        elif instruction and instruction.opcode == Opcode.XNET_S:
            print(f"{self.MAGENTA}XNET: B shifts NORTH  ^   ^   ^  (toroidal){self.RESET}")
        else:
            # Print blank line to maintain consistent spacing
            print()
        print()  # Blank line before PE arrays

    def animate_step(self, cycle: int, instruction: Instruction | None, iteration: int) -> None:
        """Animate a single simulation step."""
        self.clear_screen()
        self.print_header(cycle, instruction, iteration)

        # Always show XNET status line (blank if not an XNET instruction)
        self.print_xnet_status(instruction)

        # Show all three registers
        self.print_pe_array("A")
        self.print_pe_array("B")
        self.print_pe_array("C")

        # Handle delay/step
        if self.step_mode:
            input(f"{self.DIM}Press Enter to continue...{self.RESET}")
        elif self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

    def run(self) -> None:
        """Run the animated GEMM demonstration."""
        # Load data with Cannon's pre-skew
        self.sim.load_gemm_data(self.A, self.B, use_cannon_skew=True)

        # Generate program
        program = self.sim.create_gemm_program(self.k)
        self.sim.load_program(program)

        # Track iteration (each iteration = 4 instructions: MUL, ADD, XNET_E, XNET_S)
        iteration = 0
        instr_in_iteration = 0

        # Run with animation
        while not self.sim.done:
            # Fetch instruction (may be None if stalled)
            instr = self.sim.acu.fetch()

            if instr is not None:
                # Dispatch to PE array
                self.sim.pe_array.broadcast_instruction(instr)
                self.sim._update_statistics(instr)
                instr_in_iteration += 1

                # Update iteration counter (4 instructions per iteration)
                if instr_in_iteration >= 4:
                    iteration = min(iteration + 1, self.k - 1)
                    instr_in_iteration = 0

            # Animate
            self.animate_step(self.sim.cycle, instr, iteration)

            # Increment cycle
            self.sim.cycle += 1

            # Check completion
            if self.sim.acu.is_done():
                self.sim.done = True

        # Final display
        self.clear_screen()
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print(f"{self.GREEN}GEMM Complete!{self.RESET}")
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print()

        # Extract and verify result
        C = self.sim.extract_result(rows=self.m, cols=self.n)

        print(f"{self.BOLD}Result C:{self.RESET}")
        for row in C:
            print("  " + " ".join(f"{v:4d}" for v in row))
        print()

        print(f"{self.BOLD}Expected C:{self.RESET}")
        for row in self.expected:
            print("  " + " ".join(f"{v:4d}" for v in row))
        print()

        if np.array_equal(C, self.expected):
            print(f"{self.GREEN}Result matches expected!{self.RESET}")
        else:
            print(f"{self.RED}Result MISMATCH!{self.RESET}")

        # Statistics
        stats = self.sim.get_statistics()
        print()
        print(f"{self.BOLD}Statistics:{self.RESET}")
        print(f"  Cycles:       {stats['cycles']}")
        print(f"  Instructions: {stats['total_instructions']}")
        print(f"  ALU ops:      {stats['alu_ops']}")
        print(f"  XNET ops:     {stats['xnet_ops']}")
        print(f"  Active PEs:   {stats['active_pes']}/{stats['total_pes']}")


def main():
    parser = argparse.ArgumentParser(description="Animated MasPar SIMD GEMM visualization")
    add_animation_args(parser)
    add_gemm_args(parser, default_m=4, default_n=4, default_k=4)

    args = parser.parse_args()

    # For MasPar, array size must match matrix dimensions for Cannon's algorithm
    # Use min of m, n, k for square array
    size = min(args.m, args.n, args.k)
    if args.m != args.n or args.n != args.k:
        print(f"Note: Using {size}x{size} array (Cannon's algorithm requires square matrices)")
        args.m = args.n = args.k = size

    if should_print(args):
        print(f"MasPar SIMD {args.m}x{args.n}x{args.k} GEMM Animation")
        print(f"Array size: {args.m}x{args.n} PEs ({args.m * args.n} total)")
        print()

    animator = MasParAnimator(
        m=args.m,
        n=args.n,
        k=args.k,
        delay_ms=get_effective_delay(args),
        step_mode=args.step,
        no_color=args.no_color,
    )

    if should_print(args):
        animator.print_matrices()
        if not args.movie:
            input("Press Enter to start animation...")

    animator.run()


if __name__ == "__main__":
    main()
