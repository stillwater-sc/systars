#!/usr/bin/env python3
"""
Wavefront Animation GIF Generator.

This script generates an animated GIF showing the systolic array wavefront
during matrix multiplication. The output can be shared on Slack, in docs,
or embedded in presentations.

Usage:
    python 03_wavefront_gif.py [--size N] [--output FILE] [--fps N]

    --size N      Array size (default: 4)
    --output FILE Output filename (default: wavefront.gif)
    --fps N       Frames per second (default: 2)
    --dpi N       Image resolution (default: 100)

Requirements:
    pip install matplotlib pillow

Example:
    python 03_wavefront_gif.py --size 4 --output systolic_wavefront.gif --fps 2
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Check for matplotlib
try:
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Systolic Array Simulator (same as 02_animated_wavefront.py)
# =============================================================================


@dataclass
class PE:
    """Processing Element state."""

    a: int = 0
    b: int = 0
    c: int = 0
    a_valid: bool = False
    b_valid: bool = False
    active: bool = False


@dataclass
class SystolicArraySim:
    """Cycle-accurate systolic array simulator."""

    size: int
    A: np.ndarray = None
    B: np.ndarray = None

    pes: list = field(default_factory=list)
    cycle: int = 0
    a_queue: list = field(default_factory=list)
    b_queue: list = field(default_factory=list)

    def __post_init__(self):
        self.pes = [[PE() for _ in range(self.size)] for _ in range(self.size)]

    def setup(self, A: np.ndarray, B: np.ndarray):
        """Setup matrices and create skewed input queues."""
        self.A = A.copy()
        self.B = B.copy()
        self.cycle = 0

        for row in self.pes:
            for pe in row:
                pe.a = pe.b = pe.c = 0
                pe.a_valid = pe.b_valid = pe.active = False

        # Skewed A queues (row i delayed by i cycles)
        self.a_queue = []
        for i in range(self.size):
            row_data = [None] * i + list(A[i, :]) + [None] * (self.size - 1)
            self.a_queue.append(row_data)

        # Skewed B queues (col j delayed by j cycles)
        self.b_queue = []
        for j in range(self.size):
            col_data = [None] * j + list(B[:, j]) + [None] * (self.size - 1)
            self.b_queue.append(col_data)

    def step(self) -> bool:
        """Execute one clock cycle. Returns True if still active."""
        n = self.size

        queues_empty = all(self.cycle >= len(q) for q in self.a_queue + self.b_queue)
        any_active = any(pe.active for row in self.pes for pe in row)

        if queues_empty and not any_active:
            return False

        new_a = [None] * n
        new_b = [None] * n

        for i in range(n):
            if self.cycle < len(self.a_queue[i]):
                new_a[i] = self.a_queue[i][self.cycle]

        for j in range(n):
            if self.cycle < len(self.b_queue[j]):
                new_b[j] = self.b_queue[j][self.cycle]

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

        for i in range(n):
            for j in range(n):
                pe = self.pes[i][j]

                if j == 0:
                    if new_a[i] is not None:
                        pe.a = new_a[i]
                        pe.a_valid = True
                    else:
                        pe.a_valid = False
                else:
                    if shifted_a[i][j] is not None:
                        pe.a = shifted_a[i][j]
                        pe.a_valid = True
                    else:
                        pe.a_valid = False

                if i == 0:
                    if new_b[j] is not None:
                        pe.b = new_b[j]
                        pe.b_valid = True
                    else:
                        pe.b_valid = False
                else:
                    if shifted_b[i][j] is not None:
                        pe.b = shifted_b[i][j]
                        pe.b_valid = True
                    else:
                        pe.b_valid = False

        self.cycle += 1
        return True

    def get_result(self) -> np.ndarray:
        """Extract result matrix from PE accumulators."""
        result = np.zeros((self.size, self.size), dtype=np.int32)
        for i in range(self.size):
            for j in range(self.size):
                result[i, j] = self.pes[i][j].c
        return result


# =============================================================================
# Matplotlib Visualization
# =============================================================================


def create_animation(
    size: int = 4,
    fps: int = 2,
    dpi: int = 100,
) -> tuple:
    """
    Create matplotlib animation of systolic array wavefront.

    Returns:
        (fig, anim, sim) tuple for saving or displaying
    """
    # Create test matrices
    np.random.seed(42)
    A = np.random.randint(1, 5, size=(size, size), dtype=np.int8)
    B = np.random.randint(1, 5, size=(size, size), dtype=np.int8)
    C_expected = A.astype(np.int32) @ B.astype(np.int32)

    # Initialize simulator
    sim = SystolicArraySim(size=size)
    sim.setup(A, B)

    # Collect all states
    states = []
    states.append(capture_state(sim))
    while sim.step():
        states.append(capture_state(sim))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=dpi)
    fig.suptitle("Systolic Array Wavefront Animation", fontsize=14, fontweight="bold")

    def init():
        """Initialize animation."""
        for ax in axes:
            ax.clear()
        return []

    def animate(frame_idx):
        """Update animation frame."""
        state = states[frame_idx]
        cycle = state["cycle"]

        for ax in axes:
            ax.clear()

        # Left plot: Input matrices with current values highlighted
        ax0 = axes[0]
        ax0.set_title(f"Cycle {cycle}: Inputs", fontsize=11)

        # Show A matrix with current row highlighted
        ax0.text(
            0.5,
            0.95,
            "Matrix A (rows → right)",
            transform=ax0.transAxes,
            ha="center",
            fontsize=9,
            color="#DAA520",
        )
        ax0.text(
            0.5,
            0.45,
            "Matrix B (cols ↓ down)",
            transform=ax0.transAxes,
            ha="center",
            fontsize=9,
            color="#4169E1",
        )

        # Draw A matrix
        for i in range(size):
            for j in range(size):
                val = A[i, j]
                # Check if this element is being fed this cycle
                feeding = (
                    cycle < len(sim.a_queue[i])
                    and sim.a_queue[i][cycle] is not None
                    and j == cycle - i
                    and 0 <= cycle - i < size
                )
                color = "#FFD700" if feeding else "#FFF8DC"
                rect = patches.FancyBboxPatch(
                    (j * 0.15 + 0.1, 0.7 - i * 0.12),
                    0.12,
                    0.1,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor="#DAA520",
                    linewidth=1,
                    transform=ax0.transAxes,
                )
                ax0.add_patch(rect)
                ax0.text(
                    j * 0.15 + 0.16,
                    0.75 - i * 0.12,
                    str(val),
                    transform=ax0.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        # Draw B matrix
        for i in range(size):
            for j in range(size):
                val = B[i, j]
                feeding = (
                    cycle < len(sim.b_queue[j])
                    and sim.b_queue[j][cycle] is not None
                    and i == cycle - j
                    and 0 <= cycle - j < size
                )
                color = "#87CEEB" if feeding else "#F0F8FF"
                rect = patches.FancyBboxPatch(
                    (j * 0.15 + 0.1, 0.25 - i * 0.12),
                    0.12,
                    0.1,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor="#4169E1",
                    linewidth=1,
                    transform=ax0.transAxes,
                )
                ax0.add_patch(rect)
                ax0.text(
                    j * 0.15 + 0.16,
                    0.30 - i * 0.12,
                    str(val),
                    transform=ax0.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 1)
        ax0.axis("off")

        # Middle plot: PE array state
        ax1 = axes[1]
        ax1.set_title(f"Systolic Array ({size}×{size})", fontsize=11)

        pe_states = state["pes"]
        cell_size = 0.8 / size

        for i in range(size):
            for j in range(size):
                pe = pe_states[i][j]
                x = 0.1 + j * cell_size
                y = 0.9 - (i + 1) * cell_size

                # Determine color based on state
                if pe["active"]:
                    color = "#90EE90"  # Light green - computing
                    edge_color = "#228B22"
                elif pe["a_valid"] or pe["b_valid"]:
                    color = "#ADD8E6"  # Light blue - has data
                    edge_color = "#4169E1"
                else:
                    color = "#F5F5F5"  # Light gray - idle
                    edge_color = "#808080"

                rect = patches.FancyBboxPatch(
                    (x, y),
                    cell_size * 0.9,
                    cell_size * 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=2,
                    transform=ax1.transAxes,
                )
                ax1.add_patch(rect)

                # Cell content
                cx = x + cell_size * 0.45
                cy = y + cell_size * 0.45

                if pe["active"]:
                    text = f"{pe['a']}×{pe['b']}"
                    ax1.text(
                        cx,
                        cy + 0.02,
                        text,
                        transform=ax1.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                    )
                    ax1.text(
                        cx,
                        cy - 0.04,
                        f"c={pe['c']}",
                        transform=ax1.transAxes,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="#228B22",
                    )
                elif pe["a_valid"] and pe["b_valid"]:
                    ax1.text(
                        cx,
                        cy,
                        f"{pe['a']}×{pe['b']}",
                        transform=ax1.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                elif pe["a_valid"]:
                    ax1.text(
                        cx,
                        cy,
                        f"a={pe['a']}",
                        transform=ax1.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#DAA520",
                    )
                elif pe["b_valid"]:
                    ax1.text(
                        cx,
                        cy,
                        f"b={pe['b']}",
                        transform=ax1.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#4169E1",
                    )

        # Draw arrows for data flow
        ax1.annotate(
            "A →",
            xy=(0.02, 0.5),
            fontsize=10,
            color="#DAA520",
            fontweight="bold",
            transform=ax1.transAxes,
        )
        ax1.annotate(
            "B ↓",
            xy=(0.5, 0.98),
            fontsize=10,
            color="#4169E1",
            fontweight="bold",
            ha="center",
            transform=ax1.transAxes,
        )

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")

        # Right plot: Accumulator / Result
        ax2 = axes[2]
        ax2.set_title("Accumulator C", fontsize=11)

        c_max = max(1, max(pe["c"] for row in pe_states for pe in row))

        for i in range(size):
            for j in range(size):
                pe = pe_states[i][j]
                x = 0.15 + j * cell_size
                y = 0.85 - (i + 1) * cell_size

                # Color by value intensity
                if pe["c"] > 0:
                    intensity = min(1.0, pe["c"] / c_max)
                    color = plt.cm.Greens(0.3 + 0.7 * intensity)
                else:
                    color = "#F5F5F5"

                rect = patches.FancyBboxPatch(
                    (x, y),
                    cell_size * 0.9,
                    cell_size * 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor="#228B22" if pe["c"] > 0 else "#808080",
                    linewidth=1.5,
                    transform=ax2.transAxes,
                )
                ax2.add_patch(rect)

                cx = x + cell_size * 0.45
                cy = y + cell_size * 0.45
                ax2.text(
                    cx,
                    cy,
                    str(pe["c"]),
                    transform=ax2.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold" if pe["c"] > 0 else "normal",
                )

        # Show expected result
        ax2.text(
            0.5,
            0.08,
            "Expected:",
            transform=ax2.transAxes,
            ha="center",
            fontsize=8,
            color="#666666",
        )
        expected_str = str(C_expected.tolist()).replace("], [", "]\n[")
        ax2.text(
            0.5,
            0.02,
            expected_str,
            transform=ax2.transAxes,
            ha="center",
            fontsize=6,
            color="#666666",
            family="monospace",
        )

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")

        plt.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(states), interval=1000 // fps, blit=False
    )

    return fig, anim, sim, states


def capture_state(sim: SystolicArraySim) -> dict:
    """Capture current simulator state."""
    return {
        "cycle": sim.cycle,
        "pes": [
            [
                {
                    "a": pe.a,
                    "b": pe.b,
                    "c": pe.c,
                    "a_valid": pe.a_valid,
                    "b_valid": pe.b_valid,
                    "active": pe.active,
                }
                for pe in row
            ]
            for row in sim.pes
        ],
    }


def main():
    """Main entry point."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for GIF generation.")
        print("Install with: pip install matplotlib pillow")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate systolic array wavefront animation GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=4, help="Array size N (default: 4)")
    parser.add_argument(
        "--output",
        type=str,
        default="wavefront.gif",
        help="Output filename (default: wavefront.gif)",
    )
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    parser.add_argument("--dpi", type=int, default=100, help="Image resolution (default: 100)")
    parser.add_argument(
        "--show", action="store_true", help="Show animation in window instead of saving"
    )

    args = parser.parse_args()

    print(f"Creating {args.size}x{args.size} systolic array animation...")

    fig, anim, sim, states = create_animation(
        size=args.size,
        fps=args.fps,
        dpi=args.dpi,
    )

    print(f"Generated {len(states)} frames")

    if args.show:
        print("Showing animation (close window to exit)...")
        plt.show()
    else:
        output_path = Path(args.output)
        print(f"Saving to {output_path}...")

        # Save as GIF
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(str(output_path), writer=writer)

        print(f"Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        print("\nShare this GIF on Slack or embed in documentation!")


if __name__ == "__main__":
    main()
