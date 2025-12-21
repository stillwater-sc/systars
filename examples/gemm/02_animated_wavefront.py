#!/usr/bin/env python3
"""
Animated Systolic Array Wavefront Demo.

This example visualizes the wavefront data flow through a systolic array
during matrix multiplication C = A @ B.

The systolic array processes data in a wave pattern:
- A elements flow left-to-right, entering staggered by row
- B elements flow top-to-bottom, entering staggered by column
- Each PE performs MAC: c += a * b, then passes a right and b down

This visualization shows:
1. The skewed input patterns (how data is fed)
2. The wavefront propagation through the array
3. The accumulation of partial products
4. The final result extraction

Usage:
    python 02_animated_wavefront.py [--size N] [--delay MS] [--fast] [--step]

    --size N     Array size (default: 4)
    --delay MS   Delay between frames in milliseconds (default: 500)
    --fast       Fast mode (no animation delay)
    --step       Step mode (press Enter to advance each cycle)
    --movie      Movie mode for term2svg capture
    --no-color   Disable colored output
"""

import argparse
import os
import sys
import time

# Add parent directory to path for examples.common import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field

import numpy as np
from common.cli import add_animation_args, add_gemm_args, get_effective_delay

# =============================================================================
# ANSI Color Codes for Terminal Output
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"

    @classmethod
    def disable(cls):
        """Disable all colors."""
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def move_cursor(row: int, col: int):
    """Move cursor to position (ANSI escape)."""
    print(f"\033[{row};{col}H", end="")


# =============================================================================
# Systolic Array Simulator
# =============================================================================


@dataclass
class PE:
    """Processing Element state."""

    a: int = 0  # Current A value (flows right)
    b: int = 0  # Current B value (flows down)
    c: int = 0  # Accumulated result
    a_valid: bool = False
    b_valid: bool = False
    active: bool = False  # Currently computing


@dataclass
class SystolicArraySim:
    """
    Cycle-accurate systolic array simulator for visualization.

    Models the wavefront data flow through an NxN array computing C = A @ B.
    """

    size: int
    A: np.ndarray = None
    B: np.ndarray = None
    C: np.ndarray = None

    # Internal state
    pes: list = field(default_factory=list)
    cycle: int = 0
    a_queue: list = field(default_factory=list)  # Skewed A input queues
    b_queue: list = field(default_factory=list)  # Skewed B input queues

    def __post_init__(self):
        """Initialize PE grid and input queues."""
        # Create PE grid
        self.pes = [[PE() for _ in range(self.size)] for _ in range(self.size)]

        # Initialize result matrix
        self.C = np.zeros((self.size, self.size), dtype=np.int32)

    def setup(self, A: np.ndarray, B: np.ndarray):
        """
        Setup matrices and create skewed input queues.

        For systolic arrays, inputs are skewed:
        - Row i of A is delayed by i cycles
        - Column j of B is delayed by j cycles

        This creates the wavefront pattern.
        """
        self.A = A.copy()
        self.B = B.copy()
        self.cycle = 0

        # Reset PEs
        for row in self.pes:
            for pe in row:
                pe.a = 0
                pe.b = 0
                pe.c = 0
                pe.a_valid = False
                pe.b_valid = False
                pe.active = False

        # Create skewed A queues (one per row)
        # Row i gets i cycles of delay (None values) before its data
        # Convert to Python int to avoid int8 overflow during MAC
        self.a_queue = []
        for i in range(self.size):
            row_data = [None] * i + [int(x) for x in A[i, :]] + [None] * (self.size - 1)
            self.a_queue.append(row_data)

        # Create skewed B queues (one per column)
        # Column j gets j cycles of delay before its data
        self.b_queue = []
        for j in range(self.size):
            col_data = [None] * j + [int(x) for x in B[:, j]] + [None] * (self.size - 1)
            self.b_queue.append(col_data)

    def step(self) -> bool:
        """
        Execute one clock cycle.

        Returns True if there's still activity, False when done.
        """
        n = self.size

        # Check if we're done (all queues empty and no active PEs)
        queues_empty = all(self.cycle >= len(q) for q in self.a_queue + self.b_queue)
        any_active = any(pe.active for row in self.pes for pe in row)

        if queues_empty and not any_active:
            return False

        # Store new values to inject (we'll update after shifting)
        new_a = [None] * n
        new_b = [None] * n

        # Get new inputs from queues
        for i in range(n):
            if self.cycle < len(self.a_queue[i]):
                new_a[i] = self.a_queue[i][self.cycle]

        for j in range(n):
            if self.cycle < len(self.b_queue[j]):
                new_b[j] = self.b_queue[j][self.cycle]

        # Shift data through the array (right to left, bottom to top to avoid overwriting)
        # First, compute and shift

        # Temporary storage for shifted values
        shifted_a = [[None] * n for _ in range(n)]
        shifted_b = [[None] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                pe = self.pes[i][j]

                # Perform MAC if both inputs valid
                if pe.a_valid and pe.b_valid:
                    pe.c += pe.a * pe.b
                    pe.active = True
                else:
                    pe.active = False

                # Pass A to the right (j+1)
                if j + 1 < n and pe.a_valid:
                    shifted_a[i][j + 1] = pe.a

                # Pass B down (i+1)
                if i + 1 < n and pe.b_valid:
                    shifted_b[i + 1][j] = pe.b

        # Apply shifted values and new inputs
        for i in range(n):
            for j in range(n):
                pe = self.pes[i][j]

                # New A value: from left neighbor or input queue
                if j == 0:
                    # Leftmost column gets from input queue
                    if new_a[i] is not None:
                        pe.a = new_a[i]
                        pe.a_valid = True
                    else:
                        pe.a_valid = False
                else:
                    # Get from shifted value
                    if shifted_a[i][j] is not None:
                        pe.a = shifted_a[i][j]
                        pe.a_valid = True
                    else:
                        pe.a_valid = False

                # New B value: from top neighbor or input queue
                if i == 0:
                    # Top row gets from input queue
                    if new_b[j] is not None:
                        pe.b = new_b[j]
                        pe.b_valid = True
                    else:
                        pe.b_valid = False
                else:
                    # Get from shifted value
                    if shifted_b[i][j] is not None:
                        pe.b = shifted_b[i][j]
                        pe.b_valid = True
                    else:
                        pe.b_valid = False

        self.cycle += 1
        return True

    def get_result(self) -> np.ndarray:
        """Extract the result matrix from PE accumulators."""
        result = np.zeros((self.size, self.size), dtype=np.int32)
        for i in range(self.size):
            for j in range(self.size):
                result[i, j] = self.pes[i][j].c
        return result


# =============================================================================
# Visualization
# =============================================================================


def format_value(val: int | None, width: int = 4) -> str:
    """Format a value for display."""
    if val is None:
        return " " * width
    return f"{val:>{width}}"


def visualize_state(sim: SystolicArraySim, show_inputs: bool = True):
    """
    Print the current state of the systolic array.

    Shows:
    - Input queues (A from left, B from top)
    - PE grid with current a, b, c values
    - Active computation indicators
    """
    n = sim.size
    c = Colors

    print(f"\n{c.BOLD}{'=' * 60}{c.RESET}")
    print(f"{c.BOLD}Cycle {sim.cycle}{c.RESET}")
    print(f"{'=' * 60}\n")

    # Calculate column width based on values
    col_width = 6

    if show_inputs:
        # Show B inputs (top)
        print(f"  {c.CYAN}B inputs (flowing ↓):{c.RESET}")
        print("       ", end="")
        for j in range(n):
            idx = sim.cycle
            if idx < len(sim.b_queue[j]) and sim.b_queue[j][idx] is not None:
                val = sim.b_queue[j][idx]
                print(f"{c.CYAN}{val:^{col_width}}{c.RESET}", end=" ")
            else:
                print(f"{'·':^{col_width}}", end=" ")
        print("\n")

    # Show array with A inputs on left
    for i in range(n):
        # A input for this row
        if show_inputs:
            idx = sim.cycle
            if idx < len(sim.a_queue[i]) and sim.a_queue[i][idx] is not None:
                val = sim.a_queue[i][idx]
                print(f"{c.YELLOW}{val:>3}{c.RESET} → ", end="")
            else:
                print(f"{'·':>3}   ", end="")
        else:
            print("      ", end="")

        # PE row
        print("[", end="")
        for j in range(n):
            pe = sim.pes[i][j]

            # Determine cell color based on state
            if pe.active:
                bg = c.BG_GREEN
            elif pe.a_valid or pe.b_valid:
                bg = c.BG_BLUE
            else:
                bg = ""

            # Format cell content
            if pe.a_valid and pe.b_valid:
                # Show a*b operation
                cell = f"{pe.a}×{pe.b}"
            elif pe.a_valid:
                cell = f"a={pe.a}"
            elif pe.b_valid:
                cell = f"b={pe.b}"
            else:
                cell = "·"

            print(f"{bg}{cell:^{col_width}}{c.RESET}", end="")
            if j < n - 1:
                print(" ", end="")

        print("]")

    # Show accumulator values
    print(f"\n  {c.GREEN}Accumulator C:{c.RESET}")
    for i in range(n):
        print("       [", end="")
        for j in range(n):
            pe = sim.pes[i][j]
            if pe.c != 0:
                print(f"{c.GREEN}{pe.c:^{col_width}}{c.RESET}", end="")
            else:
                print(f"{'0':^{col_width}}", end="")
            if j < n - 1:
                print(" ", end="")
        print("]")

    print()


def visualize_skewed_inputs(A: np.ndarray, B: np.ndarray):
    """Show the skewed input pattern."""
    n = A.shape[0]
    c = Colors

    print(f"\n{c.BOLD}Input Matrices:{c.RESET}")
    print(f"\n{c.YELLOW}Matrix A (rows flow →):{c.RESET}")
    print(A)
    print(f"\n{c.CYAN}Matrix B (cols flow ↓):{c.RESET}")
    print(B)

    print(f"\n{c.BOLD}Skewed Input Pattern:{c.RESET}")
    print("(Each row/col is delayed by its index to create the wavefront)\n")

    # Show skewed A
    print(f"{c.YELLOW}Skewed A (time →):{c.RESET}")
    total_cycles = 2 * n - 1
    for i in range(n):
        print(f"  Row {i}: ", end="")
        for t in range(total_cycles + n):
            if t < i:
                print("  · ", end="")
            elif t - i < n:
                print(f"{A[i, t - i]:>3} ", end="")
            else:
                print("  · ", end="")
        print()

    print(f"\n{c.CYAN}Skewed B (time →):{c.RESET}")
    for j in range(n):
        print(f"  Col {j}: ", end="")
        for t in range(total_cycles + n):
            if t < j:
                print("  · ", end="")
            elif t - j < n:
                print(f"{B[t - j, j]:>3} ", end="")
            else:
                print("  · ", end="")
        print()


def visualize_wavefront_diagram(n: int):
    """Show a diagram of wavefront propagation."""
    c = Colors

    print(f"\n{c.BOLD}Wavefront Propagation Pattern:{c.RESET}")
    print("(Numbers show which cycle each PE first becomes active)\n")

    print("       ", end="")
    for j in range(n):
        print(f"  Col{j} ", end="")
    print()

    for i in range(n):
        print(f"  Row{i} [", end="")
        for j in range(n):
            # First active cycle is max(row_delay, col_delay) = max(i, j)
            first_active = i + j
            print(f"  {first_active:>2}  ", end="")
        print("]")

    print(f"\n  {c.DIM}Diagonal bands show the wavefront moving through the array{c.RESET}")


# =============================================================================
# Main Demo
# =============================================================================


def run_animated_demo(
    size: int = 4,
    delay_ms: int = 500,
    use_color: bool = True,
    movie_mode: bool = False,
):
    """
    Run the animated wavefront demonstration.

    Args:
        size: Array size (NxN)
        delay_ms: Milliseconds between animation frames
        use_color: Whether to use colored output
        movie_mode: If True, skip prompts and setup/summary for term2svg capture
    """
    if not use_color:
        Colors.disable()

    c = Colors

    print(f"{c.BOLD}{'=' * 60}{c.RESET}")
    print(f"{c.BOLD}Systolic Array Wavefront Animation{c.RESET}")
    print(f"{c.BOLD}{'=' * 60}{c.RESET}")

    # Create small test matrices with simple values
    np.random.seed(42)
    A = np.random.randint(1, 5, size=(size, size), dtype=np.int8)
    B = np.random.randint(1, 5, size=(size, size), dtype=np.int8)

    # Show input matrices
    visualize_skewed_inputs(A, B)

    # Show wavefront diagram
    visualize_wavefront_diagram(size)

    # Expected result
    C_expected = A.astype(np.int32) @ B.astype(np.int32)
    print(f"\n{c.BOLD}Expected Result C = A @ B:{c.RESET}")
    print(C_expected)

    # Create simulator
    sim = SystolicArraySim(size=size)
    sim.setup(A, B)

    # Prompt before starting (skip in movie mode for term2svg capture)
    if not movie_mode:
        print(f"\n{c.BOLD}Press Enter to start animation (Ctrl+C to skip)...{c.RESET}")
        try:
            input()
        except KeyboardInterrupt:
            print("\nSkipping animation...")
            return True

    # Animate
    try:
        while True:
            clear_screen()
            print(f"{c.BOLD}Systolic Array Wavefront Animation{c.RESET}")
            print(f"Computing C[{size}x{size}] = A[{size}x{size}] @ B[{size}x{size}]")

            visualize_state(sim)

            if not sim.step():
                break

            time.sleep(delay_ms / 1000.0)

    except KeyboardInterrupt:
        print("\nAnimation interrupted.")

    # Show final result
    clear_screen()
    print(f"\n{c.BOLD}{'=' * 60}{c.RESET}")
    print(f"{c.BOLD}Final State (Cycle {sim.cycle}){c.RESET}")
    print(f"{'=' * 60}")

    visualize_state(sim, show_inputs=False)

    # Verify result
    C_result = sim.get_result()

    print(f"\n{c.BOLD}Verification:{c.RESET}")
    print("\nExpected C:")
    print(C_expected)
    print("\nComputed C:")
    print(C_result)

    if np.array_equal(C_result, C_expected):
        print(f"\n{c.GREEN}{c.BOLD}✓ PASS: Result matches expected!{c.RESET}")
        return True
    else:
        print(f"\n{c.RED}{c.BOLD}✗ FAIL: Result does not match!{c.RESET}")
        return False


def run_step_by_step(size: int = 3, use_color: bool = True):
    """
    Run a step-by-step (non-animated) demonstration.

    This is useful for understanding the flow without animation.
    """
    if not use_color:
        Colors.disable()

    c = Colors

    print(f"{c.BOLD}{'=' * 60}{c.RESET}")
    print(f"{c.BOLD}Systolic Array Step-by-Step Demo{c.RESET}")
    print(f"{c.BOLD}{'=' * 60}{c.RESET}")

    # Use very simple matrices for clarity
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)[:size, :size]
    B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)[:size, :size]

    print(f"\n{c.YELLOW}Matrix A:{c.RESET}")
    print(A)
    print(f"\n{c.CYAN}Matrix B (identity for clarity):{c.RESET}")
    print(B)

    C_expected = A.astype(np.int32) @ B.astype(np.int32)
    print(f"\n{c.GREEN}Expected C = A @ B:{c.RESET}")
    print(C_expected)

    # Create simulator
    sim = SystolicArraySim(size=size)
    sim.setup(A, B)

    print(f"\n{c.BOLD}Step-by-step execution:{c.RESET}")
    print("(Press Enter for each step, 'q' to finish)\n")

    step = 0
    while True:
        visualize_state(sim)

        if not sim.step():
            print(f"{c.BOLD}Computation complete!{c.RESET}")
            break

        try:
            response = input(f"Press Enter for cycle {sim.cycle} (q to finish): ")
            if response.lower() == "q":
                # Complete remaining cycles
                while sim.step():
                    pass
                break
        except (KeyboardInterrupt, EOFError):
            break

        step += 1

    # Final verification
    C_result = sim.get_result()
    print(f"\n{c.BOLD}Final Result:{c.RESET}")
    print(C_result)

    if np.array_equal(C_result, C_expected):
        print(f"\n{c.GREEN}✓ Correct!{c.RESET}")
    else:
        print(f"\n{c.RED}✗ Mismatch!{c.RESET}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Animated Systolic Array Wavefront Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # GEMM dimensions (systolic array requires M=N for square array)
    add_gemm_args(parser, default_m=4, default_n=4, default_k=4)

    # Add common animation arguments
    add_animation_args(
        parser,
        include_max_cycles=False,  # This demo doesn't use max_cycles
    )

    args = parser.parse_args()

    # Systolic array requires square dimensions (M=N)
    if args.m != args.n:
        print(f"Error: Systolic array requires M=N (got M={args.m}, N={args.n})")
        sys.exit(1)

    array_size = args.m  # M=N for square array

    if args.step:
        run_step_by_step(size=array_size, use_color=not args.no_color)
    else:
        success = run_animated_demo(
            size=array_size,
            delay_ms=get_effective_delay(args),
            use_color=not args.no_color,
            movie_mode=args.movie,
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
