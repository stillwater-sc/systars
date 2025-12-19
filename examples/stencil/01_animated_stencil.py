#!/usr/bin/env python3
"""
Animated Stencil Machine Demo with Scratchpad and Pipelined MAC.

This example visualizes the complete data flow through a stencil machine during
2D convolution, including:
- Scratchpad memory access with wide cache line reads
- Pipelined MAC computation showing multiply → adder tree → accumulate stages

The stencil machine achieves 1× DRAM reads per input pixel (vs 9× for im2col
with 3×3 kernels) through line buffer reuse.

Data Flow:
1. Scratchpad (SRAM) stores the input image
2. Cache lines (multiple pixels) are fetched from scratchpad
3. Line buffers store K_h rows for window reuse
4. Window former extracts K_h × K_w sliding windows via shift registers
5. MAC array computes dot products:
   - MULTIPLY: K×K parallel multipliers compute all products
   - REDUCE: Adder tree sums products (log2(K×K) levels)
   - ACCUMULATE: Add to running sum for multi-channel inputs

Usage:
    python 01_animated_stencil.py [options]

    --width W        Input width (default: 16)
    --height H       Input height (default: 8)
    --kernel K       Kernel size K×K (default: 3)
    --cache-line N   Cache line width in pixels (default: 4)
    --delay MS       Delay between frames in ms (default: 300)
    --step           Single-step mode (press Enter for each cycle)
    --fast-mac       Skip MAC pipeline details (instant dot product)
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

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
    ORANGE = "\033[38;5;208m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_ORANGE = "\033[48;5;208m"

    @classmethod
    def disable(cls):
        """Disable all colors."""
        for attr in dir(cls):
            if not attr.startswith("_") and attr != "disable":
                setattr(cls, attr, "")


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


# =============================================================================
# Scratchpad Memory Model
# =============================================================================


@dataclass
class Scratchpad:
    """
    Scratchpad SRAM with wide cache line reads.

    Models a scratchpad that stores the input feature map and provides
    wide cache line reads for efficient data transfer to line buffers.
    """

    height: int
    width: int
    cache_line_width: int  # Number of pixels per cache line

    data: np.ndarray = None

    # Current read state
    read_row: int = 0
    read_col: int = 0  # Start column of current cache line
    cache_line_data: np.ndarray = None  # Current cache line contents
    cache_line_valid: bool = False

    def __post_init__(self):
        self.data = np.zeros((self.height, self.width), dtype=np.int8)
        self.cache_line_data = np.zeros(self.cache_line_width, dtype=np.int8)

    def load_image(self, image: np.ndarray):
        """Load image data into scratchpad."""
        self.data = image.copy()

    def fetch_cache_line(self, row: int, col: int) -> np.ndarray:
        """
        Fetch a cache line starting at (row, col).

        Returns cache_line_width pixels starting from col.
        Pads with zeros if reaching end of row.
        """
        self.read_row = row
        self.read_col = col

        line = np.zeros(self.cache_line_width, dtype=np.int8)
        for i in range(self.cache_line_width):
            if col + i < self.width:
                line[i] = self.data[row, col + i]
        self.cache_line_data = line
        self.cache_line_valid = True
        return line

    def get_num_cache_lines_per_row(self) -> int:
        """Return number of cache line fetches needed per row."""
        return (self.width + self.cache_line_width - 1) // self.cache_line_width


# =============================================================================
# Line Buffer with Cache Line Fill
# =============================================================================


@dataclass
class LineBuffer:
    """Single line buffer storing one row of pixels."""

    width: int
    data: np.ndarray = None
    filled: bool = False
    fill_progress: int = 0  # Number of pixels filled so far

    def __post_init__(self):
        self.data = np.zeros(self.width, dtype=np.int8)

    def write_cache_line(self, start_col: int, cache_line: np.ndarray):
        """Write a cache line to the buffer starting at start_col."""
        for i, val in enumerate(cache_line):
            if start_col + i < self.width:
                self.data[start_col + i] = val
                self.fill_progress = start_col + i + 1

        if self.fill_progress >= self.width:
            self.filled = True


# =============================================================================
# MAC Pipeline Models
# =============================================================================


@dataclass
class PipelineEntry:
    """A single entry in the pipelined MAC."""

    window_id: int  # Which window this is (for tracking)
    output_pos: tuple  # (row, col) in output
    stage: int  # Current pipeline stage
    products: list  # Per-filter products
    partial_sums: list  # Adder tree intermediate values
    final_sums: np.ndarray = None  # Final dot product results


@dataclass
class PipelinedMAC:
    """
    Pipelined MAC array with overlapped execution.

    This models a heavily pipelined MAC where:
    - Initiation Interval (II) = 1: New window every cycle
    - Latency = 1 + adder_tree_depth + 1 cycles
    - Throughput = 1 result/cycle after pipeline fills

    Pipeline stages:
      Stage 0: MULTIPLY - K×K products computed
      Stage 1..depth: REDUCE - Adder tree levels
      Stage depth+1: ACCUMULATE - Add to output

    With a 3×3 kernel (depth=4), pipeline has 6 stages.
    After 6 cycles, results emerge every cycle.
    """

    kernel_size: int
    num_filters: int

    # Pipeline configuration
    adder_tree_depth: int = 0
    total_stages: int = 0

    # Pipeline entries (one per stage, or None if empty)
    entries: list = field(default_factory=list)

    # Tracking
    window_counter: int = 0
    results_ready: list = field(default_factory=list)  # Completed entries

    # Filter coefficients
    filters: np.ndarray = None

    def __post_init__(self):
        K = self.kernel_size
        self.adder_tree_depth = math.ceil(math.log2(K * K)) if K > 1 else 0
        self.total_stages = 1 + self.adder_tree_depth + 1  # MUL + REDUCE + ACCUM
        self.entries = [None] * self.total_stages
        self.results_ready = []

    def reset(self):
        """Reset pipeline state."""
        self.entries = [None] * self.total_stages
        self.results_ready = []
        self.window_counter = 0

    def can_accept(self) -> bool:
        """Check if pipeline can accept a new window (II=1 means always)."""
        return True  # Fully pipelined

    def push(self, window: np.ndarray, output_pos: tuple):
        """Push a new window into the pipeline."""
        K = self.kernel_size

        # Compute all products immediately (stage 0)
        products = []
        for f in range(self.num_filters):
            f_products = []
            for i in range(K):
                for j in range(K):
                    product = int(window[i, j]) * int(self.filters[f, i, j])
                    f_products.append(product)
            products.append(f_products)

        entry = PipelineEntry(
            window_id=self.window_counter,
            output_pos=output_pos,
            stage=0,
            products=products,
            partial_sums=[list(p) for p in products],  # Copy for reduction
            final_sums=np.zeros(self.num_filters, dtype=np.int32),
        )
        self.window_counter += 1

        # Insert at stage 0 (push existing entry forward first)
        self._advance_pipeline()
        self.entries[0] = entry

    def _advance_pipeline(self):
        """Advance all entries through the pipeline."""
        # Check if last stage has completed entry (before moving)
        if self.entries[self.total_stages - 1] is not None:
            self.results_ready.append(self.entries[self.total_stages - 1])
            self.entries[self.total_stages - 1] = None

        # Move entries from back to front
        for stage in range(self.total_stages - 1, 0, -1):
            if self.entries[stage - 1] is not None:
                entry = self.entries[stage - 1]
                entry.stage = stage

                # Process based on stage type
                if stage <= self.adder_tree_depth:
                    # REDUCE stage: one level of adder tree
                    for f in range(self.num_filters):
                        current = entry.partial_sums[f]
                        if len(current) > 1:
                            next_level = []
                            for i in range(0, len(current), 2):
                                if i + 1 < len(current):
                                    next_level.append(current[i] + current[i + 1])
                                else:
                                    next_level.append(current[i])
                            entry.partial_sums[f] = next_level
                        elif len(current) == 1:
                            entry.final_sums[f] = current[0]

                elif stage == self.adder_tree_depth + 1:
                    # ACCUMULATE stage: finalize
                    for f in range(self.num_filters):
                        if entry.partial_sums[f]:
                            entry.final_sums[f] = entry.partial_sums[f][0]

                self.entries[stage] = entry
                self.entries[stage - 1] = None  # Clear source slot

        # Note: stage 0 is already cleared by the loop above

    def step(self) -> list:
        """
        Advance pipeline by one cycle (without pushing new entry).

        Returns list of completed results.
        """
        self._advance_pipeline()
        completed = self.results_ready.copy()
        self.results_ready = []
        return completed

    def get_occupancy(self) -> int:
        """Return number of entries in pipeline."""
        return sum(1 for e in self.entries if e is not None)

    def get_pipeline_state(self) -> list:
        """Get state of each pipeline stage for visualization."""
        states = []
        for stage, entry in enumerate(self.entries):
            if entry is None:
                states.append({"stage": stage, "occupied": False})
            else:
                stage_name = self._get_stage_name(stage)
                states.append(
                    {
                        "stage": stage,
                        "occupied": True,
                        "name": stage_name,
                        "window_id": entry.window_id,
                        "output_pos": entry.output_pos,
                        "partial_sums": [len(ps) for ps in entry.partial_sums],
                    }
                )
        return states

    def _get_stage_name(self, stage: int) -> str:
        """Get human-readable name for a pipeline stage."""
        if stage == 0:
            return "MUL"
        elif stage <= self.adder_tree_depth:
            return f"RED{stage}"
        else:
            return "ACC"


@dataclass
class MACPipeline:
    """
    Models a pipelined MAC unit for realistic dot product computation.

    Pipeline stages:
    1. MULTIPLY: K×K parallel multipliers compute products (1 cycle)
    2. REDUCE: Adder tree reduces products to single sum (log2(K×K) levels)
       - For 3×3: 9 products → 5 → 3 → 2 → 1 (4 levels)
       - For 5×5: 25 products → 13 → 7 → 4 → 2 → 1 (5 levels)
    3. ACCUMULATE: Add to running sum (1 cycle)

    In real hardware, this can be:
    - Fully combinational (small kernels, low frequency)
    - Pipelined (registers between stages for higher frequency)
    - Serialized (fewer multipliers, multiple cycles)
    """

    kernel_size: int
    num_filters: int

    # Pipeline state
    state: str = "IDLE"  # IDLE, MULTIPLY, REDUCE, ACCUMULATE, DONE
    reduce_level: int = 0  # Current adder tree level

    # Data in pipeline
    window: np.ndarray = None  # K×K input window
    filters: np.ndarray = None  # num_filters × K × K
    products: list = field(default_factory=list)  # Per-filter products
    partial_sums: list = field(default_factory=list)  # Adder tree intermediate
    final_sums: np.ndarray = None  # Final dot product results
    accumulators: np.ndarray = None  # Running accumulators

    # Configuration
    adder_tree_depth: int = 0

    def __post_init__(self):
        K = self.kernel_size
        self.adder_tree_depth = math.ceil(math.log2(K * K)) if K > 1 else 0
        self.products = [[] for _ in range(self.num_filters)]
        self.partial_sums = [[] for _ in range(self.num_filters)]
        self.final_sums = np.zeros(self.num_filters, dtype=np.int32)
        self.accumulators = np.zeros(self.num_filters, dtype=np.int32)
        self.window = np.zeros((K, K), dtype=np.int8)

    def reset(self):
        """Reset pipeline state."""
        self.state = "IDLE"
        self.reduce_level = 0
        for f in range(self.num_filters):
            self.products[f] = []
            self.partial_sums[f] = []
        self.final_sums = np.zeros(self.num_filters, dtype=np.int32)

    def clear_accumulators(self):
        """Clear accumulators for new output position."""
        self.accumulators = np.zeros(self.num_filters, dtype=np.int32)

    def start(self, window: np.ndarray, filters: np.ndarray):
        """Start MAC pipeline with new window and filters."""
        self.window = window.copy()
        self.filters = filters
        self.state = "MULTIPLY"
        self.reduce_level = 0

    def step(self) -> bool:
        """
        Execute one pipeline cycle.

        Returns True if pipeline is still active, False when done.
        """
        K = self.kernel_size

        if self.state == "IDLE" or self.state == "DONE":
            return False

        if self.state == "MULTIPLY":
            # Compute all K×K products for each filter
            for f in range(self.num_filters):
                self.products[f] = []
                for i in range(K):
                    for j in range(K):
                        product = int(self.window[i, j]) * int(self.filters[f, i, j])
                        self.products[f].append(product)
                # Initialize partial sums for reduction
                self.partial_sums[f] = self.products[f].copy()

            self.state = "REDUCE"
            self.reduce_level = 0
            return True

        if self.state == "REDUCE":
            # One level of adder tree reduction
            for f in range(self.num_filters):
                current = self.partial_sums[f]
                if len(current) <= 1:
                    # Reduction complete for this filter
                    self.final_sums[f] = current[0] if current else 0
                    continue

                # Pairwise addition
                next_level = []
                for i in range(0, len(current), 2):
                    if i + 1 < len(current):
                        next_level.append(current[i] + current[i + 1])
                    else:
                        next_level.append(current[i])  # Odd element
                self.partial_sums[f] = next_level

            self.reduce_level += 1

            # Check if all filters have reduced to single value
            all_done = all(len(self.partial_sums[f]) <= 1 for f in range(self.num_filters))
            if all_done:
                for f in range(self.num_filters):
                    if self.partial_sums[f]:
                        self.final_sums[f] = self.partial_sums[f][0]
                self.state = "ACCUMULATE"

            return True

        if self.state == "ACCUMULATE":
            # Add to accumulators
            for f in range(self.num_filters):
                self.accumulators[f] += self.final_sums[f]
            self.state = "DONE"
            return False

        return False

    def get_pipeline_info(self) -> dict:
        """Get current pipeline state for visualization."""
        return {
            "state": self.state,
            "reduce_level": self.reduce_level,
            "adder_tree_depth": self.adder_tree_depth,
            "products": [list(p) for p in self.products],
            "partial_sums": [list(p) for p in self.partial_sums],
            "final_sums": self.final_sums.tolist(),
            "accumulators": self.accumulators.tolist(),
            "window": self.window.copy(),
        }


# =============================================================================
# Stencil Machine Simulator with Scratchpad
# =============================================================================


@dataclass
class StencilMachineSim:
    """
    Cycle-accurate stencil machine simulator with scratchpad and pipelined MAC.

    Models the complete dataflow:
    1. Scratchpad → Cache line fetch
    2. Cache line → Line buffer fill
    3. Line buffers → Window former
    4. Window former → MAC array (with pipeline stages)

    States:
    - SP_FETCH: Fetching cache line from scratchpad
    - LB_FILL: Writing cache line to line buffer
    - WINDOW_SHIFT: Shifting window and reading from line buffers
    - MAC_COMPUTE: Running MAC pipeline (MULTIPLY → REDUCE → ACCUMULATE)
    - DONE: Processing complete
    """

    input_height: int
    input_width: int
    kernel_size: int
    cache_line_width: int = 4
    num_filters: int = 2
    fast_mac: bool = False  # Skip MAC pipeline details
    pipelined: bool = False  # Use pipelined MAC with II=1

    # Memory components
    scratchpad: Scratchpad = None
    line_buffers: list = field(default_factory=list)

    # MAC pipeline (one or the other based on mode)
    mac_pipeline: MACPipeline = None
    pipelined_mac: PipelinedMAC = None

    # Filter and output data
    filters: np.ndarray = None
    expected_output: np.ndarray = None
    output: np.ndarray = None

    # Window former state
    window: np.ndarray = None
    shift_regs: list = field(default_factory=list)

    # Position tracking
    cycle: int = 0
    input_row: int = 0  # Row being loaded from scratchpad
    cache_line_idx: int = 0  # Which cache line in current row
    output_row: int = 0
    output_col: int = 0
    read_col: int = 0  # Column being read from line buffers

    # Line buffer management
    lb_write_idx: int = 0  # Which line buffer to write to
    lb_read_base: int = 0  # Oldest line buffer for reading
    rows_buffered: int = 0

    # State machine
    state: str = "SP_FETCH"

    # Track if window is valid for MAC
    window_valid: bool = False
    pending_output_col: int = 0  # Output column waiting for MAC result

    # Current operation info (for visualization)
    current_cache_line: np.ndarray = None
    current_mac_results: np.ndarray = None
    last_action: str = ""

    def __post_init__(self):
        """Initialize internal structures."""
        K = self.kernel_size

        # Create scratchpad
        self.scratchpad = Scratchpad(
            height=self.input_height,
            width=self.input_width,
            cache_line_width=self.cache_line_width,
        )

        # Create K line buffers
        self.line_buffers = [LineBuffer(self.input_width) for _ in range(K)]

        # Create MAC pipeline(s)
        self.mac_pipeline = MACPipeline(
            kernel_size=K,
            num_filters=self.num_filters,
        )

        if self.pipelined:
            self.pipelined_mac = PipelinedMAC(
                kernel_size=K,
                num_filters=self.num_filters,
            )

        # Shift registers: K rows × K columns
        self.shift_regs = [[0 for _ in range(K)] for _ in range(K)]

        # Window buffer
        self.window = np.zeros((K, K), dtype=np.int8)

        # Output
        out_h = self.input_height - K + 1
        out_w = self.input_width - K + 1
        self.output = np.zeros((self.num_filters, out_h, out_w), dtype=np.int32)
        self.current_mac_results = np.zeros(self.num_filters, dtype=np.int32)
        self.current_cache_line = np.zeros(self.cache_line_width, dtype=np.int8)

    def setup(self, input_image: np.ndarray, filters: np.ndarray):
        """Setup input and filter data."""
        self.scratchpad.load_image(input_image)
        self.filters = filters.copy()

        # Compute expected output
        K = self.kernel_size
        out_h = self.input_height - K + 1
        out_w = self.input_width - K + 1
        self.expected_output = np.zeros((self.num_filters, out_h, out_w), dtype=np.int32)

        for f in range(self.num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    window = input_image[i : i + K, j : j + K]
                    self.expected_output[f, i, j] = np.sum(window * filters[f])

        # Reset state
        self.cycle = 0
        self.input_row = 0
        self.cache_line_idx = 0
        self.output_row = 0
        self.output_col = 0
        self.read_col = 0
        self.lb_write_idx = 0
        self.lb_read_base = 0
        self.rows_buffered = 0
        self.state = "SP_FETCH"
        self.window_valid = False
        self.pending_output_col = 0
        self.last_action = "Starting..."

        # Reset MAC pipeline
        self.mac_pipeline.reset()
        self.mac_pipeline.clear_accumulators()

        if self.pipelined:
            self.pipelined_mac.reset()
            self.pipelined_mac.filters = filters

        # Clear line buffers
        for lb in self.line_buffers:
            lb.data = np.zeros(self.input_width, dtype=np.int8)
            lb.filled = False
            lb.fill_progress = 0

        # Clear shift registers
        for row in range(self.kernel_size):
            for col in range(self.kernel_size):
                self.shift_regs[row][col] = 0

    def step(self) -> bool:
        """Execute one clock cycle."""
        K = self.kernel_size

        if self.state == "DONE":
            return False

        if self.state == "SP_FETCH":
            # Fetch cache line from scratchpad
            if self.input_row < self.input_height:
                start_col = self.cache_line_idx * self.cache_line_width
                cache_line = self.scratchpad.fetch_cache_line(self.input_row, start_col)
                self.current_cache_line = cache_line.copy()
                self.last_action = (
                    f"Scratchpad fetch: row {self.input_row}, "
                    f"cols [{start_col}:{start_col + self.cache_line_width}]"
                )
                self.state = "LB_FILL"

        elif self.state == "LB_FILL":
            # Write cache line to line buffer
            lb = self.line_buffers[self.lb_write_idx]
            start_col = self.cache_line_idx * self.cache_line_width
            lb.write_cache_line(start_col, self.current_cache_line)

            self.last_action = (
                f"Line buffer {self.lb_write_idx} fill: "
                f"cols [{start_col}:{start_col + self.cache_line_width}]"
            )

            # Move to next cache line or next row
            num_cache_lines = self.scratchpad.get_num_cache_lines_per_row()
            self.cache_line_idx += 1

            if self.cache_line_idx >= num_cache_lines:
                # Row complete
                self.cache_line_idx = 0
                self.input_row += 1
                self.rows_buffered += 1

                # Advance circular buffer index
                self.lb_write_idx = (self.lb_write_idx + 1) % K

                # Check if we have enough rows to start computing
                if self.rows_buffered >= K:
                    self.state = "WINDOW_SHIFT"
                    self.read_col = 0
                    self.mac_pipeline.clear_accumulators()
                else:
                    self.state = "SP_FETCH"
            else:
                self.state = "SP_FETCH"

        elif self.state == "WINDOW_SHIFT":
            # Read from line buffers and form window
            for row in range(K):
                phys_idx = (self.lb_read_base + row) % K
                lb = self.line_buffers[phys_idx]

                # Shift registers: shift left, new value enters right
                for col in range(K - 1):
                    self.shift_regs[row][col] = self.shift_regs[row][col + 1]

                # Read new value from line buffer
                if self.read_col < self.input_width:
                    self.shift_regs[row][K - 1] = lb.data[self.read_col]
                else:
                    self.shift_regs[row][K - 1] = 0

            # Copy shift registers to window
            for row in range(K):
                for col in range(K):
                    self.window[row, col] = self.shift_regs[row][col]

            # Check if window is valid for MAC
            out_w = self.input_width - K + 1
            self.window_valid = self.read_col >= K - 1 and self.output_col < out_w

            if self.window_valid:
                self.pending_output_col = self.output_col

                if self.fast_mac:
                    # Fast mode: compute instantly
                    for f in range(self.num_filters):
                        mac_result = np.sum(self.window * self.filters[f])
                        self.output[f, self.output_row, self.output_col] = mac_result
                        self.current_mac_results[f] = mac_result
                    self.last_action = (
                        f"MAC (fast): output[{self.output_row},{self.output_col}] = "
                        f"{self.current_mac_results.tolist()}"
                    )
                    self.output_col += 1
                    self.read_col += 1
                    self._check_row_complete()

                elif self.pipelined:
                    # Pipelined mode: push window to pipeline, continue immediately
                    self.pipelined_mac.push(self.window.copy(), (self.output_row, self.output_col))
                    occupancy = self.pipelined_mac.get_occupancy()
                    self.last_action = (
                        f"Pipeline push: window[{self.output_row},{self.output_col}] "
                        f"(occupancy: {occupancy}/{self.pipelined_mac.total_stages})"
                    )

                    # Check for completed results
                    for entry in self.pipelined_mac.results_ready:
                        row, col = entry.output_pos
                        for f in range(self.num_filters):
                            self.output[f, row, col] = entry.final_sums[f]
                            self.current_mac_results[f] = entry.final_sums[f]
                    self.pipelined_mac.results_ready = []

                    self.output_col += 1
                    self.read_col += 1
                    self._check_row_complete()
                else:
                    # Sequential mode: start MAC pipeline and wait
                    self.last_action = f"Window ready at col {self.read_col}, starting MAC..."
                    self.mac_pipeline.start(self.window, self.filters)
                    self.state = "MAC_COMPUTE"
            else:
                # In pipelined mode, still need to drain results while filling window
                if self.pipelined and self.pipelined_mac.get_occupancy() > 0:
                    completed = self.pipelined_mac.step()
                    for entry in completed:
                        row, col = entry.output_pos
                        for f in range(self.num_filters):
                            self.output[f, row, col] = entry.final_sums[f]
                            self.current_mac_results[f] = entry.final_sums[f]
                    self.last_action = (
                        f"Window shift: read_col={self.read_col}, "
                        f"pipeline draining ({self.pipelined_mac.get_occupancy()} in flight)"
                    )
                else:
                    self.last_action = f"Window shift: read_col={self.read_col}, filling window..."

                self.read_col += 1
                self._check_row_complete()

        elif self.state == "MAC_COMPUTE":
            # Run MAC pipeline
            mac_info = self.mac_pipeline.get_pipeline_info()
            mac_state = mac_info["state"]

            if mac_state == "MULTIPLY":
                K2 = K * K
                self.last_action = f"MAC MULTIPLY: {K2} products computing..."
                self.mac_pipeline.step()

            elif mac_state == "REDUCE":
                level = mac_info["reduce_level"]
                depth = mac_info["adder_tree_depth"]
                num_vals = len(mac_info["partial_sums"][0]) if mac_info["partial_sums"] else 0
                self.last_action = (
                    f"MAC REDUCE: Level {level + 1}/{depth}, "
                    f"{num_vals} values → {(num_vals + 1) // 2}"
                )
                self.mac_pipeline.step()

            elif mac_state == "ACCUMULATE":
                self.last_action = "MAC ACCUMULATE: Adding to output..."
                self.mac_pipeline.step()

            elif mac_state == "DONE":
                # Store result
                for f in range(self.num_filters):
                    self.output[f, self.output_row, self.pending_output_col] = (
                        self.mac_pipeline.accumulators[f]
                    )
                    self.current_mac_results[f] = self.mac_pipeline.accumulators[f]

                self.last_action = (
                    f"MAC DONE: output[{self.output_row},{self.pending_output_col}] = "
                    f"{self.current_mac_results.tolist()}"
                )

                self.mac_pipeline.reset()
                self.mac_pipeline.clear_accumulators()
                self.output_col += 1
                self.read_col += 1

                self._check_row_complete()

        elif self.state == "PIPELINE_DRAIN":
            # Drain remaining entries from pipelined MAC
            completed = self.pipelined_mac.step()
            for entry in completed:
                row, col = entry.output_pos
                for f in range(self.num_filters):
                    self.output[f, row, col] = entry.final_sums[f]
                    self.current_mac_results[f] = entry.final_sums[f]

            remaining = self.pipelined_mac.get_occupancy()
            if remaining > 0:
                self.last_action = f"Pipeline draining: {remaining} entries remaining..."
            else:
                self.state = "DONE"
                self.last_action = "Pipeline drained, processing complete!"

        self.cycle += 1
        return self.state != "DONE"

    def _check_row_complete(self):
        """Check if current output row is complete and transition states."""
        K = self.kernel_size

        if self.read_col >= self.input_width:
            # Row complete
            self.read_col = 0
            self.output_col = 0
            self.output_row += 1

            # Advance line buffer read base
            self.lb_read_base = (self.lb_read_base + 1) % K

            # Clear shift registers
            for row in range(K):
                for col in range(K):
                    self.shift_regs[row][col] = 0

            out_h = self.input_height - K + 1
            if self.output_row >= out_h:
                # Check if pipeline needs draining
                if self.pipelined and self.pipelined_mac.get_occupancy() > 0:
                    self.state = "PIPELINE_DRAIN"
                    self.last_action = (
                        f"All windows pushed, draining pipeline "
                        f"({self.pipelined_mac.get_occupancy()} remaining)..."
                    )
                else:
                    self.state = "DONE"
                    self.last_action = "Processing complete!"
            else:
                # Need to load next row
                if self.input_row < self.input_height:
                    self.state = "SP_FETCH"
                    self.last_action = "Output row complete, fetching next input row..."
                else:
                    self.state = "WINDOW_SHIFT"
        else:
            self.state = "WINDOW_SHIFT"


# =============================================================================
# Visualization
# =============================================================================


def visualize_state(sim: StencilMachineSim, compact: bool = False):
    """Print the current state of the stencil machine."""
    c = Colors
    K = sim.kernel_size

    print(f"\n{c.BOLD}{'═' * 80}{c.RESET}")
    print(f"{c.BOLD}Cycle {sim.cycle:3d} │ State: {sim.state:10s} │ {sim.last_action}{c.RESET}")
    print(f"{'═' * 80}{c.RESET}")

    # =========================================================================
    # Scratchpad Memory
    # =========================================================================
    print(
        f"\n{c.ORANGE}┌─ SCRATCHPAD (Input Feature Map: {sim.input_height}×{sim.input_width}) ─┐{c.RESET}"
    )

    # Show scratchpad with current fetch highlighted
    sp = sim.scratchpad

    # Limit display rows for compact mode
    display_rows = min(sim.input_height, 12) if compact else sim.input_height

    for i in range(display_rows):
        print(f"  {c.DIM}Row {i:2d}:{c.RESET} [", end="")
        for j in range(sim.input_width):
            val = sp.data[i, j]

            # Highlight current cache line fetch
            if sim.state == "SP_FETCH" and i == sim.input_row:
                expected_start = sim.cache_line_idx * sim.cache_line_width
                expected_end = expected_start + sim.cache_line_width
                if expected_start <= j < expected_end:
                    print(f"{c.BG_ORANGE}{c.BOLD}{val:3}{c.RESET}", end=" ")
                    continue

            # Highlight already loaded rows
            if i < sim.input_row:
                print(f"{c.DIM}{val:3}{c.RESET}", end=" ")
            else:
                print(f"{val:3}", end=" ")
        print("]")

    if display_rows < sim.input_height:
        print(f"  {c.DIM}... ({sim.input_height - display_rows} more rows){c.RESET}")

    # Show current cache line being transferred
    if sim.state in ["SP_FETCH", "LB_FILL"]:
        print(f"\n  {c.ORANGE}Cache Line [{sim.cache_line_width} pixels]:{c.RESET} [", end="")
        for val in sim.current_cache_line:
            print(f"{c.ORANGE}{c.BOLD}{val:3}{c.RESET}", end=" ")
        print(f"] → Line Buffer {sim.lb_write_idx}")

    # =========================================================================
    # Line Buffers
    # =========================================================================
    print(f"\n{c.CYAN}┌─ LINE BUFFERS ({K} rows × {sim.input_width} cols) ─┐{c.RESET}")

    for idx in range(K):
        lb = sim.line_buffers[idx]

        # Status markers
        markers = []
        if idx == sim.lb_write_idx and sim.state in ["SP_FETCH", "LB_FILL"]:
            markers.append(f"{c.YELLOW}←WRITE{c.RESET}")
        if idx == sim.lb_read_base and sim.state in ["WINDOW_SHIFT", "MAC_COMPUTE"]:
            markers.append(f"{c.GREEN}←READ{c.RESET}")

        # Fill status
        if lb.filled:
            status = f"{c.GREEN}●{c.RESET}"
        elif lb.fill_progress > 0:
            pct = lb.fill_progress * 100 // sim.input_width
            status = f"{c.YELLOW}◐{pct:2d}%{c.RESET}"
        else:
            status = f"{c.DIM}○{c.RESET}"

        marker_str = " ".join(markers)

        print(f"  LB[{idx}] {status}: [", end="")

        for j in range(sim.input_width):
            val = lb.data[j]

            # Highlight during fill (cache_line_idx was already incremented)
            if sim.state == "LB_FILL" and idx == sim.lb_write_idx:
                prev_start = (sim.cache_line_idx - 1) * sim.cache_line_width
                if prev_start >= 0 and prev_start <= j < prev_start + sim.cache_line_width:
                    print(f"{c.BG_YELLOW}{c.BOLD}{val:3}{c.RESET}", end=" ")
                    continue

            # Highlight read column during WINDOW_SHIFT/MAC_COMPUTE
            is_read_buffer = idx in [(sim.lb_read_base + r) % K for r in range(K)]
            is_computing = sim.state in ["WINDOW_SHIFT", "MAC_COMPUTE"]
            if is_computing and j == sim.read_col - 1 and is_read_buffer:
                print(f"{c.BG_CYAN}{val:3}{c.RESET}", end=" ")
                continue

            if lb.filled or j < lb.fill_progress:
                print(f"{c.CYAN}{val:3}{c.RESET}", end=" ")
            else:
                print(f"{c.DIM}{val:3}{c.RESET}", end=" ")

        print(f"] {marker_str}")

    # =========================================================================
    # Window Former
    # =========================================================================
    print(f"\n{c.MAGENTA}┌─ WINDOW FORMER (Shift Registers {K}×{K}) ─┐{c.RESET}")

    # Show shift registers with data flow arrow
    is_computing = sim.state in ["WINDOW_SHIFT", "MAC_COMPUTE"]
    window_valid = sim.read_col >= K and is_computing

    for row in range(K):
        print(f"  Row {row}: ", end="")
        # Show shift direction
        print(f"{c.DIM}←{c.RESET} [", end="")
        for col in range(K):
            val = sim.shift_regs[row][col]
            if val != 0:
                print(f"{c.MAGENTA}{c.BOLD}{val:3}{c.RESET}", end=" ")
            else:
                print(f"{c.DIM}{val:3}{c.RESET}", end=" ")
        print("]")

    if window_valid:
        print(f"  {c.GREEN}Window VALID → MAC{c.RESET}")
    else:
        print(f"  {c.DIM}Window filling...{c.RESET}")

    # =========================================================================
    # Filters & MAC Pipeline
    # =========================================================================
    print(f"\n{c.BLUE}┌─ FILTERS ({sim.num_filters} × {K}×{K}) ─┐{c.RESET}")

    for f in range(sim.num_filters):
        # Show filter coefficients compactly
        filter_str = " ".join(f"{sim.filters[f, r, col]:2d}" for r in range(K) for col in range(K))
        print(f"  F{f}: [{filter_str}]", end="")

        # Show MAC result
        if sim.state in ["WINDOW_SHIFT", "MAC_COMPUTE"] and sim.output_col > 0:
            result = sim.current_mac_results[f]
            print(f" → {c.GREEN}{c.BOLD}{result:4d}{c.RESET}")
        else:
            print()

    # =========================================================================
    # MAC Pipeline Visualization (always show to reserve screen space)
    # =========================================================================
    mac_info = sim.mac_pipeline.get_pipeline_info()
    mac_state = mac_info["state"]
    depth = mac_info["adder_tree_depth"]
    K2 = K * K

    # Header always visible
    if sim.pipelined:
        occupancy = sim.pipelined_mac.get_occupancy()
        total = sim.pipelined_mac.total_stages
        print(f"\n{c.RED}┌─ PIPELINED MAC (II=1, {occupancy}/{total} in-flight) ─┐{c.RESET}")
    elif sim.fast_mac:
        print(f"\n{c.RED}┌─ MAC PIPELINE (fast mode - combinational) ─┐{c.RESET}")
    else:
        print(f"\n{c.RED}┌─ MAC PIPELINE (Stage: {mac_state:10s}) ─┐{c.RESET}")

    # Pipeline stage visualization (fixed height: 4 lines)
    if sim.pipelined:
        # Pipelined mode: show pipeline stages with occupancy
        pipe_state = sim.pipelined_mac.get_pipeline_state()
        total = sim.pipelined_mac.total_stages

        # Build stage header
        stage_names = []
        for s in pipe_state:
            if s["occupied"]:
                stage_names.append(f"{c.YELLOW}{s['name']}{c.RESET}")
            else:
                stage_names.append(
                    f"{c.DIM}{sim.pipelined_mac._get_stage_name(s['stage'])}{c.RESET}"
                )
        print(f"  Stages: {' → '.join(stage_names)}")

        print(f"  {c.DIM}{'─' * 50}{c.RESET}")

        # Show which windows are in each stage
        in_flight = [(s["stage"], s["output_pos"], s["name"]) for s in pipe_state if s["occupied"]]
        if in_flight:
            entries_str = ", ".join(f"{name}@({r},{col})" for _, (r, col), name in in_flight[:4])
            print(f"  In-flight: {c.YELLOW}{entries_str}{c.RESET}")
        else:
            print(f"  {c.DIM}In-flight: (empty){c.RESET}")

        # Show last result if available
        if sim.output_col > 0 or sim.output_row > 0:
            print(f"  Last out: {c.GREEN}{sim.current_mac_results[:2].tolist()}{c.RESET}")
        else:
            print(f"  {c.DIM}Last out: (none){c.RESET}")

    elif sim.fast_mac:
        # Fast mode: show single-cycle combinational
        print(f"  MULTIPLY → REDUCE → ACCUM  {c.DIM}(1 cycle){c.RESET}")
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        if sim.state in ["WINDOW_SHIFT", "MAC_COMPUTE"] and sim.output_col > 0:
            for f in range(min(sim.num_filters, 2)):
                print(f"    F{f}: {c.GREEN}{sim.current_mac_results[f]:6d}{c.RESET}")
        else:
            for f in range(min(sim.num_filters, 2)):
                print(f"    F{f}: {c.DIM}  ----{c.RESET}")

    elif mac_state == "IDLE":
        print(f"  {c.DIM}MULTIPLY → REDUCE(×{depth}) → ACCUM{c.RESET}")
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        print(f"  {c.DIM}Waiting for window...{c.RESET}")
        print(f"  {c.DIM}                     {c.RESET}")

    elif mac_state == "MULTIPLY":
        print(f"  {c.YELLOW}▶ MULTIPLY{c.RESET} → REDUCE(×{depth}) → ACCUM")
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        print(f"  {c.YELLOW}Computing {K2} products...{c.RESET}")
        print(f"  {c.DIM}                     {c.RESET}")

    elif mac_state == "REDUCE":
        level = mac_info["reduce_level"]
        print(f"  MULTIPLY → {c.YELLOW}▶ REDUCE {level + 1}/{depth}{c.RESET} → ACCUM")
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        # Show partial sums compactly
        partial = mac_info["partial_sums"][0] if mac_info["partial_sums"] else []
        if len(partial) <= 8:
            vals = " ".join(f"{v:5d}" for v in partial)
            print(f"  F0: [{vals}]")
        else:
            print(f"  F0: [{len(partial)} vals → {(len(partial) + 1) // 2}]")
        print(f"  {c.DIM}                     {c.RESET}")

    elif mac_state == "ACCUMULATE":
        print(f"  MULTIPLY → REDUCE(×{depth}) → {c.GREEN}▶ ACCUM{c.RESET}")
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        for f in range(min(sim.num_filters, 2)):
            final = mac_info["final_sums"][f]
            accum = mac_info["accumulators"][f]
            print(f"  F{f}: {final:5d} + acc → {c.GREEN}{accum:5d}{c.RESET}")

    elif mac_state == "DONE":
        print(f"  MULTIPLY → REDUCE(×{depth}) → ACCUM → {c.GREEN}▶ OUT{c.RESET}")
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        for f in range(min(sim.num_filters, 2)):
            accum = mac_info["accumulators"][f]
            print(f"  F{f}: {c.GREEN}{c.BOLD}{accum:6d}{c.RESET} → output")

    else:
        # Fallback for any other state
        print(f"  {c.DIM}{'─' * 40}{c.RESET}")
        print(f"  {c.DIM}                     {c.RESET}")
        print(f"  {c.DIM}                     {c.RESET}")

    # =========================================================================
    # Output
    # =========================================================================
    out_h = sim.input_height - K + 1
    out_w = sim.input_width - K + 1
    print(f"\n{c.GREEN}┌─ OUTPUT ({sim.num_filters} × {out_h}×{out_w}) ─┐{c.RESET}")

    for f in range(sim.num_filters):
        print(f"  Ch{f}: [", end="")
        for row in range(out_h):
            if row > 0:
                print("\n        [", end="")
            for col in range(out_w):
                val = sim.output[f, row, col]

                # Highlight last computed position
                is_current = (
                    row == sim.output_row
                    and col == sim.output_col - 1
                    and sim.state in ["WINDOW_SHIFT", "MAC_COMPUTE"]
                )

                if is_current:
                    print(f"{c.BG_GREEN}{c.BOLD}{val:4d}{c.RESET}", end=" ")
                elif val != 0:
                    print(f"{c.GREEN}{val:4d}{c.RESET}", end=" ")
                else:
                    print(f"{c.DIM}{val:4d}{c.RESET}", end=" ")
            print("]", end="")
        print()

    print()


def visualize_architecture():
    """Print a diagram of the stencil machine architecture with scratchpad."""
    c = Colors

    print(f"""
{c.BOLD}Stencil Machine Architecture with Scratchpad{c.RESET}
{c.BOLD}============================================={c.RESET}

{c.ORANGE}SCRATCHPAD{c.RESET}        {c.CYAN}LINE BUFFERS{c.RESET}        {c.MAGENTA}WINDOW FORMER{c.RESET}      {c.GREEN}MAC ARRAY{c.RESET}
  (SRAM)            (SRAM)            (Shift Regs)       (Compute)

┌──────────┐      ┌─────────────┐      ┌───────────┐     ┌───────────┐
│ Feature  │      │   LB[0]     │      │ ┌─┬─┬─┐   │     │ Filter 0  │──→ Out[0]
│   Map    │ ═══> │   Row 0     │ ───┐ │ └─┴─┴─┘   │ ──→ ├───────────┤
│          │      ├─────────────┤    │ │ ┌─┬─┬─┐   │     │ Filter 1  │──→ Out[1]
│  H × W   │      │   LB[1]     │ ───┼→│ └─┴─┴─┘   │     ├───────────┤
│ pixels   │      │   Row 1     │    │ │ ┌─┬─┬─┐   │     │    ...    │
│          │      ├─────────────┤    │ │ └─┴─┴─┘   │     └───────────┘
│          │      │   LB[K-1]   │ ───┘ │   K×K     │
└──────────┘      │   Row K-1   │      │  Window   │     {c.DIM}Accumulate{c.RESET}
     │            └─────────────┘      └───────────┘     {c.DIM}across input{c.RESET}
     │                  ↑                                {c.DIM}channels{c.RESET}
     └──────────────────┘
       {c.ORANGE}Cache Line{c.RESET}
       {c.ORANGE}(N pixels){c.RESET}

{c.BOLD}Data Flow:{c.RESET}
  1. {c.ORANGE}Scratchpad{c.RESET} stores entire input feature map
  2. {c.ORANGE}Cache lines{c.RESET} (N pixels) fetched per cycle → efficient SRAM access
  3. {c.CYAN}Line buffers{c.RESET} store K_h rows → enables {c.CYAN}row reuse{c.RESET}
  4. {c.MAGENTA}Shift registers{c.RESET} extract sliding window → enables {c.MAGENTA}column reuse{c.RESET}
  5. {c.GREEN}MAC array{c.RESET} computes P_c filters in parallel

{c.BOLD}Key Benefits:{c.RESET}
  • {c.GREEN}1× DRAM reads per pixel{c.RESET} (vs 9× for im2col with 3×3 kernels)
  • Wide cache line reads maximize SRAM bandwidth utilization
  • Line buffers amortize memory latency across sliding windows
""")


# =============================================================================
# Main Demo
# =============================================================================


def run_animated_demo(
    width: int = 16,
    height: int = 8,
    kernel_size: int = 3,
    cache_line_width: int = 4,
    num_filters: int = 2,
    delay_ms: int = 300,
    step_mode: bool = False,
    use_color: bool = True,
    fast_mac: bool = False,
    pipelined: bool = False,
):
    """Run the animated stencil machine demonstration."""
    if not use_color:
        Colors.disable()

    c = Colors
    K = kernel_size

    print(f"{c.BOLD}{'═' * 80}{c.RESET}")
    print(f"{c.BOLD}Stencil Machine Animation with Scratchpad{c.RESET}")
    print(f"{c.BOLD}{'═' * 80}{c.RESET}")

    # Show architecture
    visualize_architecture()

    # Create test data
    np.random.seed(42)
    input_image = np.random.randint(1, 10, size=(height, width), dtype=np.int8)
    filters = np.random.randint(-2, 3, size=(num_filters, K, K), dtype=np.int8)

    print(f"\n{c.ORANGE}Input Image ({height}×{width}):{c.RESET}")
    print(input_image)

    print(f"\n{c.BLUE}Filters ({num_filters} × {K}×{K}):{c.RESET}")
    for f in range(num_filters):
        print(f"  Filter {f}:\n{filters[f]}")

    # Create simulator
    sim = StencilMachineSim(
        input_height=height,
        input_width=width,
        kernel_size=kernel_size,
        cache_line_width=cache_line_width,
        num_filters=num_filters,
        fast_mac=fast_mac,
        pipelined=pipelined,
    )
    sim.setup(input_image, filters)

    print(f"\n{c.GREEN}Expected Output:{c.RESET}")
    for f in range(num_filters):
        print(f"  Channel {f}:\n{sim.expected_output[f]}")

    print(f"\n{c.BOLD}Configuration:{c.RESET}")
    print(f"  Cache line width: {cache_line_width} pixels")
    print(f"  Cache lines per row: {sim.scratchpad.get_num_cache_lines_per_row()}")

    if step_mode:
        print(f"\n{c.BOLD}STEP MODE: Press Enter to advance, 'q' to quit, 'r' to run{c.RESET}")
    else:
        print(f"\n{c.BOLD}Press Enter to start animation (Ctrl+C to stop)...{c.RESET}")

    try:
        input()
    except KeyboardInterrupt:
        print("\nSkipping animation...")
        return True

    # Animation loop
    running = not step_mode  # If step mode, start paused
    try:
        while True:
            clear_screen()
            print(f"{c.BOLD}Stencil Machine Animation{c.RESET}")
            print(f"Conv2D: Input [{height}×{width}] ⊛ Filter [{K}×{K}] → Output")
            if step_mode:
                print(f"{c.DIM}[STEP MODE] Enter=next, r=run, q=quit{c.RESET}")

            visualize_state(sim, compact=height > 10)

            if sim.state == "DONE":
                break

            # Handle input
            if step_mode and not running:
                try:
                    user_input = input(f"{c.BOLD}>{c.RESET} ").strip().lower()
                    if user_input == "q":
                        break
                    elif user_input == "r":
                        running = True
                        print(f"{c.GREEN}Running...{c.RESET}")
                except EOFError:
                    break
            else:
                time.sleep(delay_ms / 1000.0)

            if not sim.step():
                break

    except KeyboardInterrupt:
        print("\nAnimation interrupted.")

    # Show final result
    clear_screen()
    print(f"\n{c.BOLD}{'═' * 80}{c.RESET}")
    print(f"{c.BOLD}Final State (Cycle {sim.cycle}){c.RESET}")
    print(f"{'═' * 80}")

    visualize_state(sim, compact=False)

    # Verify result
    print(f"\n{c.BOLD}Verification:{c.RESET}")
    for f in range(num_filters):
        print(f"\n  Filter {f}:")
        print(f"    Expected: {sim.expected_output[f].tolist()}")
        print(f"    Computed: {sim.output[f].tolist()}")

        if np.array_equal(sim.output[f], sim.expected_output[f]):
            print(f"    {c.GREEN}✓ MATCH{c.RESET}")
        else:
            print(f"    {c.RED}✗ MISMATCH{c.RESET}")

    all_match = all(
        np.array_equal(sim.output[f], sim.expected_output[f]) for f in range(num_filters)
    )

    if all_match:
        print(f"\n{c.GREEN}{c.BOLD}✓ PASS: All outputs match expected!{c.RESET}")
        return True
    else:
        print(f"\n{c.RED}{c.BOLD}✗ FAIL: Some outputs don't match!{c.RESET}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Animated Stencil Machine Demo with Scratchpad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--width", type=int, default=16, help="Input width (default: 16)")
    parser.add_argument("--height", type=int, default=8, help="Input height (default: 8)")
    parser.add_argument("--kernel", type=int, default=3, help="Kernel size K×K (default: 3)")
    parser.add_argument(
        "--cache-line", type=int, default=4, help="Cache line width in pixels (default: 4)"
    )
    parser.add_argument("--filters", type=int, default=2, help="Number of filters P_c (default: 2)")
    parser.add_argument(
        "--delay", type=int, default=300, help="Delay between frames in ms (default: 300)"
    )
    parser.add_argument(
        "--step", action="store_true", help="Single-step mode (press Enter for each cycle)"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--fast-mac", action="store_true", help="Skip MAC pipeline details (instant dot product)"
    )
    parser.add_argument(
        "--pipelined",
        action="store_true",
        help="Use pipelined MAC with II=1 (high throughput mode)",
    )

    args = parser.parse_args()

    success = run_animated_demo(
        width=args.width,
        height=args.height,
        kernel_size=args.kernel,
        cache_line_width=args.cache_line,
        num_filters=args.filters,
        delay_ms=args.delay,
        step_mode=args.step,
        use_color=not args.no_color,
        fast_mac=args.fast_mac,
        pipelined=args.pipelined,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
