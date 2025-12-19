#!/usr/bin/env python3
"""
Stencil Machine Animation GIF Generator.

This script generates an animated GIF showing the stencil machine dataflow
during 2D convolution. The output can be shared on Slack, in docs,
or embedded in presentations.

The animation shows:
1. Input image with sliding window highlight
2. Line buffer state (K_h rows stored)
3. Window former extracting K×K window
4. MAC array computing dot products with filters
5. Output accumulation

Usage:
    python 02_stencil_gif.py [--width W] [--height H] [--kernel K] [--output FILE]

    --width W     Input width (default: 6)
    --height H    Input height (default: 5)
    --kernel K    Kernel size K×K (default: 3)
    --output FILE Output filename (default: stencil.gif)
    --fps N       Frames per second (default: 2)

Requirements:
    pip install matplotlib pillow numpy
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import matplotlib.animation as animation
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Stencil Machine Simulator
# =============================================================================


@dataclass
class LineBuffer:
    """Single line buffer storing one row of pixels."""

    width: int
    data: np.ndarray = None
    filled: bool = False

    def __post_init__(self):
        self.data = np.zeros(self.width, dtype=np.int8)


@dataclass
class StencilMachineSim:
    """Cycle-accurate stencil machine simulator."""

    input_height: int
    input_width: int
    kernel_size: int
    num_filters: int = 2

    input_image: np.ndarray = None
    filters: np.ndarray = None
    expected_output: np.ndarray = None

    line_buffers: list = field(default_factory=list)
    shift_regs: list = field(default_factory=list)
    window: np.ndarray = None

    cycle: int = 0
    input_row: int = 0
    input_col: int = 0
    output_row: int = 0
    output_col: int = 0
    read_col: int = 0

    lb_write_idx: int = 0
    lb_read_base: int = 0
    rows_buffered: int = 0

    output: np.ndarray = None
    state: str = "FILLING"

    def __post_init__(self):
        K = self.kernel_size
        self.line_buffers = [LineBuffer(self.input_width) for _ in range(K)]
        self.shift_regs = [[0 for _ in range(K)] for _ in range(K)]
        self.window = np.zeros((K, K), dtype=np.int8)

        out_h = self.input_height - K + 1
        out_w = self.input_width - K + 1
        self.output = np.zeros((self.num_filters, out_h, out_w), dtype=np.int32)

    def setup(self, input_image: np.ndarray, filters: np.ndarray):
        """Setup input and filter data."""
        self.input_image = input_image.copy()
        self.filters = filters.copy()

        K = self.kernel_size
        out_h = self.input_height - K + 1
        out_w = self.input_width - K + 1
        self.expected_output = np.zeros((self.num_filters, out_h, out_w), dtype=np.int32)

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    window = input_image[i : i + K, j : j + K]
                    self.expected_output[f, i, j] = np.sum(window * filters[f])

        self.cycle = 0
        self.input_row = 0
        self.input_col = 0
        self.output_row = 0
        self.output_col = 0
        self.read_col = 0
        self.lb_write_idx = 0
        self.lb_read_base = 0
        self.rows_buffered = 0
        self.state = "FILLING"

        for lb in self.line_buffers:
            lb.data = np.zeros(self.input_width, dtype=np.int8)
            lb.filled = False

        for row in range(self.kernel_size):
            for col in range(self.kernel_size):
                self.shift_regs[row][col] = 0

    def step(self) -> bool:
        """Execute one clock cycle."""
        K = self.kernel_size

        if self.state == "DONE":
            return False

        if self.state == "FILLING":
            if self.input_row < self.input_height:
                lb = self.line_buffers[self.lb_write_idx]
                lb.data[self.input_col] = self.input_image[self.input_row, self.input_col]

                self.input_col += 1

                if self.input_col >= self.input_width:
                    lb.filled = True
                    self.input_col = 0
                    self.input_row += 1
                    self.rows_buffered += 1
                    self.lb_write_idx = (self.lb_write_idx + 1) % K

                    if self.rows_buffered >= K:
                        self.state = "COMPUTING"
                        self.output_col = 0

        elif self.state == "COMPUTING":
            for row in range(K):
                phys_idx = (self.lb_read_base + row) % K
                lb = self.line_buffers[phys_idx]

                for col in range(K - 1):
                    self.shift_regs[row][col] = self.shift_regs[row][col + 1]

                # Read new value from line buffer (sequential from column 0)
                if self.read_col < self.input_width:
                    self.shift_regs[row][K - 1] = lb.data[self.read_col]
                else:
                    self.shift_regs[row][K - 1] = 0

            for row in range(K):
                for col in range(K):
                    self.window[row, col] = self.shift_regs[row][col]

            # Compute MAC for all filters (only after window is full)
            out_w = self.input_width - K + 1
            if self.read_col >= K - 1 and self.output_col < out_w:
                for f in range(self.num_filters):
                    mac_result = np.sum(self.window * self.filters[f])
                    self.output[f, self.output_row, self.output_col] = mac_result
                self.output_col += 1

            # Advance read column
            self.read_col += 1

            if self.read_col >= self.input_width:
                self.read_col = 0
                self.output_col = 0
                self.output_row += 1
                self.lb_read_base = (self.lb_read_base + 1) % K

                for row in range(K):
                    for col in range(K):
                        self.shift_regs[row][col] = 0

                out_h = self.input_height - K + 1
                if self.output_row >= out_h:
                    self.state = "DONE"
                else:
                    if self.input_row < self.input_height:
                        self.state = "FILLING"

        self.cycle += 1
        return self.state != "DONE"


# =============================================================================
# Matplotlib Visualization
# =============================================================================


def capture_state(sim: StencilMachineSim) -> dict:
    """Capture current simulator state."""
    return {
        "cycle": sim.cycle,
        "state": sim.state,
        "input_row": sim.input_row,
        "input_col": sim.input_col,
        "output_row": sim.output_row,
        "output_col": sim.output_col,
        "read_col": sim.read_col,
        "lb_write_idx": sim.lb_write_idx,
        "lb_read_base": sim.lb_read_base,
        "rows_buffered": sim.rows_buffered,
        "line_buffers": [{"data": lb.data.copy(), "filled": lb.filled} for lb in sim.line_buffers],
        "shift_regs": [row[:] for row in sim.shift_regs],
        "window": sim.window.copy(),
        "output": sim.output.copy(),
    }


def create_animation(
    width: int = 6,
    height: int = 5,
    kernel_size: int = 3,
    num_filters: int = 2,
    fps: int = 2,
    dpi: int = 100,
) -> tuple:
    """Create matplotlib animation of stencil machine."""
    K = kernel_size

    # Create test data
    np.random.seed(42)
    input_image = np.random.randint(1, 5, size=(height, width), dtype=np.int8)
    filters = np.random.randint(-2, 3, size=(num_filters, K, K), dtype=np.int8)

    # Initialize simulator
    sim = StencilMachineSim(
        input_height=height,
        input_width=width,
        kernel_size=kernel_size,
        num_filters=num_filters,
    )
    sim.setup(input_image, filters)

    # Collect all states
    states = []
    states.append(capture_state(sim))
    while sim.step():
        states.append(capture_state(sim))

    # Create figure with 4 panels
    fig = plt.figure(figsize=(16, 10), dpi=dpi)
    fig.suptitle("Stencil Machine Dataflow Animation", fontsize=14, fontweight="bold")

    # Create grid: 2x2 layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1.2], height_ratios=[1, 1])

    ax_input = fig.add_subplot(gs[0, 0])  # Input image
    ax_linebuf = fig.add_subplot(gs[0, 1])  # Line buffers
    ax_window = fig.add_subplot(gs[0, 2])  # Window former + filters
    ax_output = fig.add_subplot(gs[1, :])  # Output

    def animate(frame_idx):
        """Update animation frame."""
        state = states[frame_idx]
        cycle = state["cycle"]
        sim_state = state["state"]

        for ax in [ax_input, ax_linebuf, ax_window, ax_output]:
            ax.clear()

        # =====================================================================
        # Panel 1: Input Image with Window Highlight
        # =====================================================================
        ax_input.set_title(f"Cycle {cycle}: Input Image", fontsize=11)

        # Draw input image cells
        cell_w = 0.12
        cell_h = 0.12
        margin = 0.05

        for i in range(height):
            for j in range(width):
                val = input_image[i, j]
                x = margin + j * cell_w
                y = 1 - margin - (i + 1) * cell_h

                # Determine cell color
                # Window is at columns [read_col - K + 1, read_col] when read_col >= K-1
                window_col_start = state["read_col"] - K + 1
                in_window = (
                    sim_state == "COMPUTING"
                    and state["read_col"] >= K - 1
                    and state["output_row"] <= i < state["output_row"] + K
                    and window_col_start <= j <= state["read_col"]
                )
                is_current_input = (
                    sim_state == "FILLING" and i == state["input_row"] and j == state["input_col"]
                )

                if is_current_input:
                    color = "#FFD700"  # Gold - current input
                    edge_color = "#DAA520"
                    edge_width = 3
                elif in_window:
                    color = "#87CEEB"  # Light blue - in window
                    edge_color = "#4169E1"
                    edge_width = 2
                else:
                    color = "#F5F5F5"
                    edge_color = "#808080"
                    edge_width = 1

                rect = patches.FancyBboxPatch(
                    (x, y),
                    cell_w * 0.9,
                    cell_h * 0.9,
                    boxstyle="round,pad=0.01",
                    facecolor=color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    transform=ax_input.transAxes,
                )
                ax_input.add_patch(rect)
                ax_input.text(
                    x + cell_w * 0.45,
                    y + cell_h * 0.45,
                    str(val),
                    transform=ax_input.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold" if in_window or is_current_input else "normal",
                )

        # Add legend
        ax_input.text(
            0.5,
            0.02,
            f"State: {sim_state}",
            transform=ax_input.transAxes,
            ha="center",
            fontsize=9,
            style="italic",
        )

        ax_input.set_xlim(0, 1)
        ax_input.set_ylim(0, 1)
        ax_input.axis("off")

        # =====================================================================
        # Panel 2: Line Buffers
        # =====================================================================
        ax_linebuf.set_title(f"Line Buffers ({K} rows)", fontsize=11)

        lb_cell_w = 0.8 / width
        lb_cell_h = 0.2

        for idx in range(K):
            lb = state["line_buffers"][idx]
            y = 0.8 - idx * (lb_cell_h + 0.05)

            # Label
            is_write = idx == state["lb_write_idx"]
            is_oldest = idx == state["lb_read_base"]
            label_color = "#DAA520" if is_write else ("#228B22" if is_oldest else "#808080")
            label = f"LB[{idx}]"
            if is_write:
                label += " ←W"
            if is_oldest:
                label += " ←R"

            ax_linebuf.text(
                0.02,
                y + lb_cell_h * 0.4,
                label,
                transform=ax_linebuf.transAxes,
                fontsize=8,
                color=label_color,
            )

            for j in range(width):
                val = lb["data"][j]
                x = 0.15 + j * lb_cell_w

                color = "#90EE90" if lb["filled"] else "#F5F5F5"
                edge = "#228B22" if lb["filled"] else "#808080"

                rect = patches.Rectangle(
                    (x, y),
                    lb_cell_w * 0.9,
                    lb_cell_h * 0.8,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=1,
                    transform=ax_linebuf.transAxes,
                )
                ax_linebuf.add_patch(rect)
                ax_linebuf.text(
                    x + lb_cell_w * 0.45,
                    y + lb_cell_h * 0.4,
                    str(val),
                    transform=ax_linebuf.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax_linebuf.text(
            0.5,
            0.02,
            f"Rows buffered: {state['rows_buffered']}",
            transform=ax_linebuf.transAxes,
            ha="center",
            fontsize=8,
        )

        ax_linebuf.set_xlim(0, 1)
        ax_linebuf.set_ylim(0, 1)
        ax_linebuf.axis("off")

        # =====================================================================
        # Panel 3: Window Former + Filters
        # =====================================================================
        ax_window.set_title("Window → MAC", fontsize=11)

        # Draw window (shift registers)
        win_cell = 0.08
        win_x_start = 0.05
        win_y_start = 0.75

        ax_window.text(
            win_x_start,
            win_y_start + K * win_cell + 0.03,
            "Window (K×K):",
            transform=ax_window.transAxes,
            fontsize=9,
            fontweight="bold",
        )

        for i in range(K):
            for j in range(K):
                val = state["shift_regs"][i][j]
                x = win_x_start + j * win_cell
                y = win_y_start + (K - 1 - i) * win_cell

                color = "#E6E6FA" if val != 0 else "#F5F5F5"
                rect = patches.Rectangle(
                    (x, y),
                    win_cell * 0.9,
                    win_cell * 0.9,
                    facecolor=color,
                    edgecolor="#9370DB",
                    linewidth=1.5,
                    transform=ax_window.transAxes,
                )
                ax_window.add_patch(rect)
                ax_window.text(
                    x + win_cell * 0.45,
                    y + win_cell * 0.45,
                    str(val),
                    transform=ax_window.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold" if val != 0 else "normal",
                )

        # Draw filters
        filt_x_start = 0.45
        filt_y_start = 0.55

        for f_idx in range(num_filters):
            fy = filt_y_start - f_idx * (K * win_cell + 0.15)
            ax_window.text(
                filt_x_start,
                fy + K * win_cell + 0.02,
                f"Filter {f_idx}:",
                transform=ax_window.transAxes,
                fontsize=8,
            )

            for i in range(K):
                for j in range(K):
                    val = filters[f_idx, i, j]
                    x = filt_x_start + j * win_cell
                    y = fy + (K - 1 - i) * win_cell

                    if val > 0:
                        color = "#90EE90"
                    elif val < 0:
                        color = "#FFB6C1"
                    else:
                        color = "#F5F5F5"

                    rect = patches.Rectangle(
                        (x, y),
                        win_cell * 0.9,
                        win_cell * 0.9,
                        facecolor=color,
                        edgecolor="#4169E1",
                        linewidth=1,
                        transform=ax_window.transAxes,
                    )
                    ax_window.add_patch(rect)
                    ax_window.text(
                        x + win_cell * 0.45,
                        y + win_cell * 0.45,
                        str(val),
                        transform=ax_window.transAxes,
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

            # Show MAC result - display most recently computed output
            if (
                sim_state == "COMPUTING" and state["output_col"] > 0  # At least one output computed
            ):
                # Show the most recently computed output (output_col - 1)
                result = state["output"][f_idx, state["output_row"], state["output_col"] - 1]
                ax_window.text(
                    filt_x_start + K * win_cell + 0.08,
                    fy + K * win_cell / 2,
                    f"→ {result}",
                    transform=ax_window.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    color="#228B22",
                )

        ax_window.set_xlim(0, 1)
        ax_window.set_ylim(0, 1)
        ax_window.axis("off")

        # =====================================================================
        # Panel 4: Output
        # =====================================================================
        out_h = height - K + 1
        out_w = width - K + 1

        ax_output.set_title(f"Output ({num_filters} filters × {out_h}×{out_w})", fontsize=11)

        out_cell = 0.06
        out_margin = 0.1

        for f_idx in range(num_filters):
            fx = out_margin + f_idx * (out_w * out_cell + 0.15)

            ax_output.text(
                fx, 0.85, f"Filter {f_idx} output:", transform=ax_output.transAxes, fontsize=9
            )

            # Find max value for color scaling
            max_val = max(1, np.abs(state["output"][f_idx]).max())

            for i in range(out_h):
                for j in range(out_w):
                    val = state["output"][f_idx, i, j]
                    x = fx + j * out_cell
                    y = 0.75 - (i + 1) * out_cell

                    # Color based on value
                    if val > 0:
                        intensity = min(1.0, abs(val) / max_val)
                        color = plt.cm.Greens(0.3 + 0.5 * intensity)
                    elif val < 0:
                        intensity = min(1.0, abs(val) / max_val)
                        color = plt.cm.Reds(0.3 + 0.5 * intensity)
                    else:
                        color = "#F5F5F5"

                    # Highlight the most recently computed output position
                    is_current = (
                        sim_state == "COMPUTING"
                        and state["output_col"] > 0
                        and i == state["output_row"]
                        and j == state["output_col"] - 1
                    )

                    rect = patches.FancyBboxPatch(
                        (x, y),
                        out_cell * 0.9,
                        out_cell * 0.9,
                        boxstyle="round,pad=0.01",
                        facecolor=color,
                        edgecolor="#FFD700" if is_current else "#808080",
                        linewidth=3 if is_current else 1,
                        transform=ax_output.transAxes,
                    )
                    ax_output.add_patch(rect)
                    ax_output.text(
                        x + out_cell * 0.45,
                        y + out_cell * 0.45,
                        str(val) if val != 0 else "0",
                        transform=ax_output.transAxes,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold" if val != 0 else "normal",
                    )

        # Expected output
        ax_output.text(
            0.5,
            0.15,
            "Expected:",
            transform=ax_output.transAxes,
            ha="center",
            fontsize=9,
            color="#666666",
        )
        expected_str = f"Filter 0: {sim.expected_output[0].tolist()}  |  Filter 1: {sim.expected_output[1].tolist()}"
        ax_output.text(
            0.5,
            0.08,
            expected_str,
            transform=ax_output.transAxes,
            ha="center",
            fontsize=7,
            color="#666666",
            family="monospace",
        )

        ax_output.set_xlim(0, 1)
        ax_output.set_ylim(0, 1)
        ax_output.axis("off")

        plt.tight_layout()
        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(states), interval=1000 // fps, blit=False
    )

    return fig, anim, sim, states


def main():
    """Main entry point."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for GIF generation.")
        print("Install with: pip install matplotlib pillow")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Generate stencil machine animation GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=6, help="Input width (default: 6)")
    parser.add_argument("--height", type=int, default=5, help="Input height (default: 5)")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size K×K (default: 3)")
    parser.add_argument("--filters", type=int, default=2, help="Number of filters (default: 2)")
    parser.add_argument(
        "--output", type=str, default="stencil.gif", help="Output filename (default: stencil.gif)"
    )
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    parser.add_argument("--dpi", type=int, default=100, help="Image resolution (default: 100)")
    parser.add_argument("--show", action="store_true", help="Show animation instead of saving")

    args = parser.parse_args()

    print("Creating stencil machine animation...")
    print(f"  Input: {args.height}×{args.width}")
    print(f"  Kernel: {args.kernel}×{args.kernel}")
    print(f"  Filters: {args.filters}")

    fig, anim, sim, states = create_animation(
        width=args.width,
        height=args.height,
        kernel_size=args.kernel,
        num_filters=args.filters,
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

        writer = animation.PillowWriter(fps=args.fps)
        anim.save(str(output_path), writer=writer)

        print(f"Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        print("\nShare this GIF on Slack or embed in documentation!")


if __name__ == "__main__":
    main()
