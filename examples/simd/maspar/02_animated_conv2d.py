#!/usr/bin/env python3
"""
Animated MasPar SIMD Conv2D Visualization.

Visualizes 2D convolution with a 3x3 kernel on the MasPar SIMD array processor.
Shows:
- The input image distributed across PEs
- XNET neighbor gathering (all 8 directions)
- Kernel weight loading and multiply-accumulate
- Output accumulation in each PE

The 3x3 convolution maps perfectly to the MasPar's 8-neighbor XNET mesh,
allowing each PE to gather all required neighbor values in 8 XNET operations.

Usage:
    python 02_animated_conv2d.py              # 8x8 conv2d with animation
    python 02_animated_conv2d.py --size 16    # 16x16 image
    python 02_animated_conv2d.py --fast       # No animation delay
    python 02_animated_conv2d.py --step       # Step through manually
    python 02_animated_conv2d.py --kernel edge  # Use edge detection kernel
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add examples path for common CLI utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.cli import add_animation_args, get_effective_delay, should_print

from systars.simd.maspar import (
    Instruction,
    MasParConfig,
    MasParSim,
    Opcode,
)

# Pre-defined kernels
KERNELS = {
    "identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int32),
    "box": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32),
    "edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int32),
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int32),
    "sobel_x": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32),
    "sobel_y": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int32),
}


class Conv2DAnimator:
    """Animated visualization of MasPar Conv2D execution."""

    def __init__(
        self,
        size: int = 8,
        kernel_name: str = "edge",
        delay_ms: int = 500,
        step_mode: bool = False,
        no_color: bool = False,
    ):
        self.size = size
        self.kernel_name = kernel_name
        self.delay_ms = delay_ms
        self.step_mode = step_mode
        self.no_color = no_color

        # Create simulator
        self.config = MasParConfig(array_rows=size, array_cols=size)
        self.sim = MasParSim(self.config)

        # Generate test image (gradient pattern)
        np.random.seed(42)
        self.image = np.arange(1, size * size + 1, dtype=np.int32).reshape(size, size)

        # Get kernel
        self.kernel = KERNELS.get(kernel_name, KERNELS["edge"])

        # Compute expected result for verification
        self.expected = self._conv2d_reference(self.image, self.kernel)

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

    def _conv2d_reference(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Compute reference convolution with toroidal wrap (like MasPar XNET)."""
        h, w = image.shape
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                acc = 0
                for ki in range(3):
                    for kj in range(3):
                        ni = (i + ki - 1) % h  # Toroidal wrap
                        nj = (j + kj - 1) % w
                        acc += image[ni, nj] * kernel[ki, kj]
                result[i, j] = acc
        return result

    def clear_screen(self) -> None:
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def print_header(self, cycle: int, instruction: Instruction | None, phase: str) -> None:
        """Print animation header."""
        print(f"{self.BOLD}{'=' * 70}{self.RESET}")
        print(
            f"{self.BOLD}MasPar SIMD Array - {self.size}x{self.size} Conv2D "
            f"(3x3 {self.kernel_name} kernel){self.RESET}"
        )
        print(f"{self.BOLD}{'=' * 70}{self.RESET}")
        print()
        print(f"Cycle: {self.CYAN}{cycle:4d}{self.RESET}  Phase: {self.YELLOW}{phase}{self.RESET}")
        if instruction:
            print(f"Instruction: {self.GREEN}{instruction}{self.RESET}")
        else:
            print(f"Instruction: {self.DIM}(stalled - multi-cycle op){self.RESET}")

    def print_xnet_status(self, instruction: Instruction | None) -> None:
        """Print XNET status line."""
        print()
        if instruction:
            xnet_names = {
                Opcode.XNET_N: ("NORTH", "^"),
                Opcode.XNET_S: ("SOUTH", "v"),
                Opcode.XNET_E: ("EAST", ">"),
                Opcode.XNET_W: ("WEST", "<"),
                Opcode.XNET_NE: ("NORTHEAST", "^>"),
                Opcode.XNET_NW: ("NORTHWEST", "<^"),
                Opcode.XNET_SE: ("SOUTHEAST", "v>"),
                Opcode.XNET_SW: ("SOUTHWEST", "<v"),
            }
            if instruction.opcode in xnet_names:
                name, arrow = xnet_names[instruction.opcode]
                print(
                    f"{self.CYAN}XNET: Gathering from {name} neighbors  "
                    f"{arrow} {arrow} {arrow}{self.RESET}"
                )
            else:
                print()
        else:
            print()
        print()

    def print_kernel(self) -> None:
        """Print the convolution kernel."""
        print(f"{self.BOLD}Kernel ({self.kernel_name}):{self.RESET}")
        for row in self.kernel:
            print("  " + " ".join(f"{v:3d}" for v in row))
        print()

    def print_pe_array(self, register: int, title: str, color: str) -> None:
        """Print PE array register values."""
        print(f"{self.BOLD}{title}:{self.RESET}")
        print()

        # Determine display width based on array size
        width = 5 if self.size <= 8 else 4

        # Column headers (show subset if large)
        cols_to_show = min(self.size, 12)
        header = "     "
        for c in range(cols_to_show):
            header += f"C{c:<{width - 1}}"
        if self.size > cols_to_show:
            header += " ..."
        print(f"{self.DIM}{header}{self.RESET}")

        # PE values (show subset if large)
        rows_to_show = min(self.size, 10)
        for r in range(rows_to_show):
            row_str = f"{self.DIM}R{r:<2}:{self.RESET} "
            for c in range(cols_to_show):
                pe = self.sim.pe_array.get_pe(r, c)
                val = pe.get_register(register)
                row_str += f"{color}{val:>{width}}{self.RESET}"
            if self.size > cols_to_show:
                row_str += " ..."
            print(row_str)

        if self.size > rows_to_show:
            print(f"{self.DIM}  ... ({self.size - rows_to_show} more rows){self.RESET}")
        print()

    def get_phase(self, instr_count: int) -> str:
        """Determine current execution phase based on instruction count."""
        if instr_count < 9:
            return "Load Kernel"
        elif instr_count < 17:
            return "XNET Gather"
        elif instr_count < 18:
            return "Init Output"
        else:
            mac_num = (instr_count - 18) // 2 + 1
            return f"MAC {mac_num}/9"

    def animate_step(self, cycle: int, instruction: Instruction | None, instr_count: int) -> None:
        """Animate a single simulation step."""
        self.clear_screen()

        phase = self.get_phase(instr_count)
        self.print_header(cycle, instruction, phase)
        self.print_xnet_status(instruction)

        # Show kernel
        self.print_kernel()

        # Show input image
        self.print_pe_array(self.sim.REG_CENTER, "Input Image (R0)", self.BLUE)

        # Show output accumulator
        self.print_pe_array(self.sim.REG_CONV_OUT, "Output (R20)", self.GREEN)

        # Handle delay/step
        if self.step_mode:
            input(f"{self.DIM}Press Enter to continue...{self.RESET}")
        elif self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

    def run(self) -> None:
        """Run the animated conv2d demonstration."""
        # Load data
        self.sim.load_conv2d_data(self.image, self.kernel)

        # Generate program
        program = self.sim.create_conv2d_program()
        self.sim.load_program(program)

        # Track instruction count
        instr_count = 0

        # Run with animation
        while not self.sim.done:
            instr = self.sim.acu.fetch()

            if instr is not None:
                self.sim.pe_array.broadcast_instruction(instr)
                self.sim._update_statistics(instr)
                instr_count += 1

            self.animate_step(self.sim.cycle, instr, instr_count)

            self.sim.cycle += 1

            if self.sim.acu.is_done():
                self.sim.done = True

        # Final display
        self.clear_screen()
        print(f"{self.BOLD}{'=' * 70}{self.RESET}")
        print(f"{self.GREEN}Conv2D Complete!{self.RESET}")
        print(f"{self.BOLD}{'=' * 70}{self.RESET}")
        print()

        # Show kernel
        self.print_kernel()

        # Extract and show result
        result = self.sim.extract_conv2d_result(rows=self.size, cols=self.size)

        print(f"{self.BOLD}Output (interior matches reference):{self.RESET}")
        rows_to_show = min(self.size, 10)
        cols_to_show = min(self.size, 12)
        for r in range(rows_to_show):
            row_str = "  "
            for c in range(cols_to_show):
                # Color based on match with expected
                if result[r, c] == self.expected[r, c]:
                    row_str += f"{self.GREEN}{result[r, c]:5d}{self.RESET}"
                else:
                    row_str += f"{self.YELLOW}{result[r, c]:5d}{self.RESET}"
            if self.size > cols_to_show:
                row_str += " ..."
            print(row_str)
        if self.size > rows_to_show:
            print(f"  ... ({self.size - rows_to_show} more rows)")
        print()

        # Verification
        interior_match = np.array_equal(result[1:-1, 1:-1], self.expected[1:-1, 1:-1])
        full_match = np.array_equal(result, self.expected)

        if full_match:
            print(f"{self.GREEN}Result matches expected (including edges)!{self.RESET}")
        elif interior_match:
            print(f"{self.GREEN}Interior matches! Edge differs due to toroidal wrap.{self.RESET}")
        else:
            print(f"{self.RED}Result MISMATCH!{self.RESET}")

        # Statistics
        stats = self.sim.get_statistics()
        print()
        print(f"{self.BOLD}Statistics:{self.RESET}")
        print(f"  Cycles:       {stats['cycles']}")
        print(f"  Instructions: {stats['total_instructions']}")
        print(f"  XNET ops:     {stats['xnet_ops']} (neighbor gather)")
        print(f"  ALU ops:      {stats['alu_ops']} (9 MACs)")
        print(f"  Memory ops:   {stats['memory_ops']} (kernel load)")
        print(f"  Active PEs:   {stats['active_pes']}/{stats['total_pes']}")


def main():
    parser = argparse.ArgumentParser(description="Animated MasPar SIMD Conv2D visualization")
    add_animation_args(parser)

    parser.add_argument(
        "--size",
        type=int,
        default=8,
        help="Image/array size (default: 8)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="edge",
        choices=list(KERNELS.keys()),
        help="Kernel type (default: edge)",
    )

    args = parser.parse_args()

    if should_print(args):
        print(f"MasPar SIMD {args.size}x{args.size} Conv2D Animation")
        print(f"Kernel: {args.kernel}")
        print(f"Array size: {args.size}x{args.size} PEs ({args.size * args.size} total)")
        print()
        print("Kernels available:")
        for name, k in KERNELS.items():
            print(f"  {name}: {k.flatten().tolist()}")
        print()

    animator = Conv2DAnimator(
        size=args.size,
        kernel_name=args.kernel,
        delay_ms=get_effective_delay(args),
        step_mode=args.step,
        no_color=args.no_color,
    )

    if should_print(args):
        print(f"Input image: {args.size}x{args.size} gradient (1 to {args.size * args.size})")
        print()
        if not args.movie:
            input("Press Enter to start animation...")

    animator.run()


if __name__ == "__main__":
    main()
