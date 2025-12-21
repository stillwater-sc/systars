#!/usr/bin/env python3
"""
Animated SIMT Streaming Multiprocessor Visualization.

This example demonstrates the NVIDIA-style Streaming Multiprocessor (SM)
architecture with real-time animation showing:
- Warp scheduling and issue
- Register file bank accesses and conflicts
- Operand collection progress
- Execution pipeline flow
- Energy consumption breakdown

The visualization helps understand why SIMT architectures have significant
instruction overhead compared to systolic arrays and stencil machines.

Usage:
    python 01_animated_simt.py [options]

Options:
    --warps N       Number of warps to simulate (auto-calculated for tiled)
    --instructions  Number of instructions per warp (default: 16)
    --delay MS      Delay between frames in milliseconds (default: 200)
    --fast          Fast mode (no animation delay)
    --step          Step mode (press Enter to advance each cycle)
    --gemm          Use GEMM workload (sequential, same program in all warps)
    --tiled         Use tiled GEMM (parallel, each warp computes different tile)
    -m M            GEMM output rows (default: 8)
    -n N            GEMM output cols (default: 8)
    --k K           GEMM reduction dimension (default: 4)

Examples:
    # Basic test program
    python 01_animated_simt.py --fast

    # Sequential GEMM (4 warps running same program, uses MOV not LD)
    python 01_animated_simt.py --gemm --k 4 --fast

    # Tiled GEMM: 8x8 output = 64 elements = 2 warps (uses real LD/ST)
    # Note: --fast-mem reduces memory latency from 200 to 4 cycles for demo
    python 01_animated_simt.py --tiled -m 8 -n 8 --k 2 --fast --fast-mem

    # Tiled GEMM step mode: 8x4 output = 32 elements = 1 warp
    python 01_animated_simt.py --tiled -m 8 -n 4 --k 2 --step --fast-mem
"""

import argparse
import re
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 3)[0] + "/src")

from systars.simt import (
    CollectorState,
    SIMTConfig,
    SMSim,
    SMState,
    WarpState,
    create_gemm_program,
    create_test_program,
    create_tiled_gemm_program,
    get_gemm_warp_count,
    get_warp_tile_info,
)


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Bright/intense colors for warp color coding
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Warp colors (cycle through these)
    WARP_COLORS = [
        "\033[94m",  # Bright blue
        "\033[92m",  # Bright green
        "\033[93m",  # Bright yellow
        "\033[95m",  # Bright magenta
        "\033[96m",  # Bright cyan
        "\033[91m",  # Bright red
        "\033[97m",  # Bright white
        "\033[33m",  # Yellow
    ]


class GEMMTracker:
    """
    Track GEMM computation state for visualization.

    Shows which output matrix elements are being computed by which warps,
    and flashes when partial sums are updated.
    """

    def __init__(self, M: int, N: int, K: int, warp_size: int = 32):
        self.M = M
        self.N = N
        self.K = K
        self.warp_size = warp_size

        # Track partial sum progress per element (how many k iterations done)
        # partial_sums[row][col] = count of FMAs completed
        self.partial_sums = [[0] * N for _ in range(M)]

        # Track which elements were updated THIS cycle (for flashing)
        self.updated_this_cycle = set()  # Set of (row, col) tuples

        # Track which elements have been stored (ST completed)
        # stored[row][col] = True if result written to memory
        self.stored = [[False] * N for _ in range(M)]

        # Track pending stores per warp
        self.pending_stores: set[int] = set()  # Set of warp_ids with pending ST

        # Track which warp owns which element
        # Element (row, col) -> linear index -> warp_id
        self.total_elements = M * N
        self.num_warps = (self.total_elements + warp_size - 1) // warp_size

    def get_warp_for_element(self, row: int, col: int) -> int:
        """Get which warp owns this output element."""
        linear_idx = row * self.N + col
        return linear_idx // self.warp_size

    def get_thread_for_element(self, row: int, col: int) -> int:
        """Get which thread within the warp computes this element."""
        linear_idx = row * self.N + col
        return linear_idx % self.warp_size

    def record_fma_completion(self, warp_id: int, thread_mask: int = 0xFFFFFFFF):
        """
        Record that an FMA completed for threads in the given warp.

        This updates the partial sum count for all elements owned by this warp.
        """
        base_element = warp_id * self.warp_size
        for thread_id in range(self.warp_size):
            if thread_mask & (1 << thread_id):
                element_idx = base_element + thread_id
                if element_idx < self.total_elements:
                    row = element_idx // self.N
                    col = element_idx % self.N
                    self.partial_sums[row][col] += 1
                    self.updated_this_cycle.add((row, col))

    def clear_updates(self):
        """Clear the updated flags for next cycle."""
        self.updated_this_cycle.clear()

    def is_element_complete(self, row: int, col: int) -> bool:
        """Check if element has completed all K iterations."""
        return self.partial_sums[row][col] >= self.K

    def get_element_progress(self, row: int, col: int) -> float:
        """Get progress of element as fraction 0.0 to 1.0."""
        return min(1.0, self.partial_sums[row][col] / self.K)

    def record_store_issued(self, warp_id: int):
        """Record that a store was issued for this warp."""
        self.pending_stores.add(warp_id)

    def record_store_complete(self, warp_id: int):
        """Record that a store completed for this warp."""
        self.pending_stores.discard(warp_id)
        # Mark all elements owned by this warp as stored
        base_element = warp_id * self.warp_size
        for thread_id in range(self.warp_size):
            element_idx = base_element + thread_id
            if element_idx < self.total_elements:
                row = element_idx // self.N
                col = element_idx % self.N
                self.stored[row][col] = True

    def is_element_stored(self, row: int, col: int) -> bool:
        """Check if element has been stored to memory."""
        return self.stored[row][col]

    def get_pending_store_count(self) -> int:
        """Get count of warps with pending stores."""
        return len(self.pending_stores)

    def is_truly_complete(self) -> bool:
        """Check if all elements are computed AND stored."""
        return all(self.stored[r][c] for r in range(self.M) for c in range(self.N))


class TimelineLogger:
    """
    Log issue and complete events for timeline analysis.

    Creates a detailed log of when each operation is issued and completes,
    helping identify bubbles and serialization issues.

    Supports two output formats:
    - csv: Simple CSV format for spreadsheet analysis
    - chrome: Chrome Trace format (JSON) for chrome://tracing or Perfetto
    """

    def __init__(self, filename: str = "timeline.log"):
        self.filename = filename
        self.events: list[dict] = []
        self.enabled = False
        self.format = "csv"
        # Track in-flight events for Chrome Trace format (to compute duration)
        # Key: (partition, warp, opcode, dst) -> issue event
        self.in_flight: dict[tuple, dict] = {}
        # Chrome Trace events (complete events with duration)
        self.trace_events: list[dict] = []

    def enable(self, filename: str = "timeline.log", fmt: str = "csv"):
        """
        Enable timeline logging to specified file.

        Args:
            filename: Output file path
            fmt: Output format - "csv" or "chrome" (Chrome Trace JSON)
        """
        self.filename = filename
        self.format = fmt
        self.enabled = True
        self.events = []
        self.in_flight = {}
        self.trace_events = []

        if self.format == "csv":
            # Write CSV header
            with open(self.filename, "w") as f:
                f.write("cycle,event,partition,warp,opcode,dst,src1,src2,latency,info\n")
        # Chrome format is written at finalize()

    def log_issue(
        self,
        cycle: int,
        partition: int,
        warp: int,
        opcode: str,
        dst: int,
        src1: int,
        src2: int,
        latency: int,
        info: str = "",
    ):
        """Log an instruction issue event."""
        if not self.enabled:
            return
        event = {
            "cycle": cycle,
            "event": "ISSUE",
            "partition": partition,
            "warp": warp,
            "opcode": opcode,
            "dst": dst,
            "src1": src1,
            "src2": src2,
            "latency": latency,
            "info": info,
        }
        self.events.append(event)

        if self.format == "csv":
            self._write_csv_event(event)
        else:
            # Track in-flight for duration calculation
            key = (partition, warp, opcode, dst)
            self.in_flight[key] = event

    def log_complete(
        self, cycle: int, partition: int, warp: int, opcode: str, dst: int, info: str = ""
    ):
        """Log an instruction completion event."""
        if not self.enabled:
            return
        event = {
            "cycle": cycle,
            "event": "COMPLETE",
            "partition": partition,
            "warp": warp,
            "opcode": opcode,
            "dst": dst,
            "src1": -1,
            "src2": -1,
            "latency": 0,
            "info": info,
        }
        self.events.append(event)

        if self.format == "csv":
            self._write_csv_event(event)
        else:
            # Match with in-flight issue event
            key = (partition, warp, opcode, dst)
            issue_event = self.in_flight.pop(key, None)
            if issue_event:
                self._add_chrome_trace_event(issue_event, cycle)
            else:
                # No matching issue - create instant event at completion
                self._add_chrome_trace_instant(cycle, partition, warp, opcode, dst, info)

    def log_stall(self, cycle: int, partition: int, warp: int, reason: str):
        """Log a stall event."""
        if not self.enabled:
            return
        event = {
            "cycle": cycle,
            "event": "STALL",
            "partition": partition,
            "warp": warp,
            "opcode": "",
            "dst": -1,
            "src1": -1,
            "src2": -1,
            "latency": 0,
            "info": reason,
        }
        self.events.append(event)

        if self.format == "csv":
            self._write_csv_event(event)
        else:
            # Stalls are instant events in Chrome Trace
            self._add_chrome_trace_stall(cycle, partition, warp, reason)

    def _write_csv_event(self, event: dict):
        """Append event to CSV log file."""
        with open(self.filename, "a") as f:
            f.write(
                f"{event['cycle']},{event['event']},{event['partition']},"
                f"{event['warp']},{event['opcode']},{event['dst']},"
                f"{event['src1']},{event['src2']},{event['latency']},"
                f'"{event["info"]}"\n'
            )

    def _add_chrome_trace_event(self, issue_event: dict, complete_cycle: int):
        """Add a complete Chrome Trace event with duration."""
        issue_cycle = issue_event["cycle"]
        duration = complete_cycle - issue_cycle

        # Categorize by opcode type
        opcode = issue_event["opcode"]
        if opcode in ("LD", "ST"):
            category = "memory"
        elif opcode in ("FFMA", "FMA", "IMUL", "MUL"):
            category = "compute,mul"
        elif opcode in ("IADD", "ADD", "SUB", "MOV"):
            category = "compute,alu"
        else:
            category = "compute"

        # Chrome Trace event format
        # ts/dur in microseconds - treat 1 cycle = 1 ns (realistic for ~1 GHz GPU)
        # Chrome Trace uses µs, so we keep cycles as-is (1 cycle = 1 µs for display)
        # This makes 200-cycle memory latency display as 200 µs = 0.2 ms
        # Real HBM3: ~100 ns, GDDR6: ~150-200 ns at ~1 GHz clock
        trace_event = {
            "name": opcode,
            "cat": category,
            "ph": "X",  # Complete event (has duration)
            "ts": float(issue_cycle),  # microseconds (1 cycle = 1 µs for readability)
            "dur": float(max(1, duration)),  # duration in microseconds
            "pid": issue_event["partition"],  # Partition as process
            "tid": issue_event["warp"],  # Warp as thread
            "args": {
                "dst": issue_event["dst"],
                "src1": issue_event["src1"],
                "src2": issue_event["src2"],
                "latency_cycles": issue_event["latency"],
                "duration_cycles": duration,
            },
        }
        self.trace_events.append(trace_event)

    def _add_chrome_trace_instant(
        self, cycle: int, partition: int, warp: int, opcode: str, dst: int, info: str
    ):
        """Add an instant Chrome Trace event (no matching issue)."""
        trace_event = {
            "name": f"{opcode}_complete",
            "cat": "memory" if opcode in ("LD", "ST") else "compute",
            "ph": "i",  # Instant event
            "ts": float(cycle),  # 1 cycle = 1 µs
            "s": "t",  # Scope: thread
            "pid": partition,
            "tid": warp,
            "args": {"dst": dst, "info": info},
        }
        self.trace_events.append(trace_event)

    def _add_chrome_trace_stall(self, cycle: int, partition: int, warp: int, reason: str):
        """Add a stall as an instant Chrome Trace event."""
        trace_event = {
            "name": "STALL",
            "cat": "stall",
            "ph": "i",  # Instant event
            "ts": float(cycle),  # 1 cycle = 1 µs
            "s": "t",  # Scope: thread
            "pid": partition,
            "tid": warp,
            "args": {"reason": reason},
        }
        self.trace_events.append(trace_event)

    def finalize(self):
        """
        Finalize the trace file (required for Chrome Trace format).

        Call this at the end of simulation to write the JSON file.
        """
        if not self.enabled or self.format != "chrome":
            return

        import json

        # Build Chrome Trace JSON structure
        trace_data = {
            "traceEvents": self.trace_events,
            "displayTimeUnit": "ns",
            "metadata": {
                "description": "SIMT Streaming Multiprocessor Simulation",
                "time_scaling": "1 cycle = 1 µs (1000x stretched for visibility)",
                "real_timing": "At 1 GHz, 1 cycle = 1 ns. Divide displayed times by 1000 for real values.",
                "example": "200 µs displayed = 200 cycles = 200 ns real (at 1 GHz)",
            },
            # Process and thread names for better visualization
            "stackFrames": {},
        }

        # Add process/thread name metadata events
        partitions_seen: set[int] = set()
        warps_seen: set[tuple[int, int]] = set()
        for e in self.trace_events:
            pid = e.get("pid", 0)
            tid = e.get("tid", 0)
            if pid not in partitions_seen:
                partitions_seen.add(pid)
                trace_data["traceEvents"].insert(
                    0,
                    {
                        "name": "process_name",
                        "ph": "M",
                        "pid": pid,
                        "args": {"name": f"Partition {pid}"},
                    },
                )
            if (pid, tid) not in warps_seen:
                warps_seen.add((pid, tid))
                trace_data["traceEvents"].insert(
                    0,
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"name": f"Warp {tid}"},
                    },
                )

        with open(self.filename, "w") as f:
            json.dump(trace_data, f, indent=2)

        print(f"Chrome Trace written to: {self.filename}")
        print("  Open in Chrome: chrome://tracing or https://ui.perfetto.dev")

    def print_summary(self):
        """Print summary of logged events."""
        if not self.events:
            return

        # Finalize Chrome Trace file if needed
        if self.format == "chrome":
            self.finalize()

        issues = [e for e in self.events if e["event"] == "ISSUE"]
        completes = [e for e in self.events if e["event"] == "COMPLETE"]
        stalls = [e for e in self.events if e["event"] == "STALL"]

        print(f"\nTimeline Summary ({self.filename}):")
        print(f"  Format: {self.format}")
        print(f"  Total events: {len(self.events)}")
        print(f"  Issues: {len(issues)}")
        print(f"  Completes: {len(completes)}")
        print(f"  Stalls: {len(stalls)}")

        # Count by opcode
        opcode_counts: dict[str, int] = {}
        for e in issues:
            op = e["opcode"]
            opcode_counts[op] = opcode_counts.get(op, 0) + 1
        print(f"  By opcode: {opcode_counts}")


# Global timeline logger
timeline_logger = TimelineLogger()


def clear_screen():
    """Clear terminal screen and move cursor to top-left."""
    print("\033[2J\033[H", end="")


def move_cursor(row: int, col: int):
    """Move cursor to specified position."""
    print(f"\033[{row};{col}H", end="")


def draw_box(title: str, width: int = 70) -> str:
    """Draw a box header."""
    padding = width - len(title) - 5
    return f"┌─ {title} " + "─" * padding + "┐"


def draw_box_end(width: int = 70) -> str:
    """Draw a box footer."""
    return "└" + "─" * (width - 2) + "┘"


def format_energy(energy_pj: float) -> str:
    """Format energy value with appropriate unit."""
    if energy_pj >= 1e6:
        return f"{energy_pj / 1e6:.2f} µJ"
    elif energy_pj >= 1e3:
        return f"{energy_pj / 1e3:.2f} nJ"
    else:
        return f"{energy_pj:.1f} pJ"


def format_warp_state(state: WarpState) -> str:
    """Format warp state with color."""
    state_colors = {
        WarpState.INACTIVE: Colors.DIM + "·" + Colors.RESET,
        WarpState.READY: Colors.GREEN + "R" + Colors.RESET,
        WarpState.ISSUED: Colors.YELLOW + "I" + Colors.RESET,
        WarpState.EXECUTING: Colors.CYAN + "E" + Colors.RESET,
        WarpState.STALLED_RAW: Colors.RED + "W" + Colors.RESET,
        WarpState.STALLED_BANK: Colors.RED + "B" + Colors.RESET,
        WarpState.STALLED_MEM: Colors.MAGENTA + "M" + Colors.RESET,
        WarpState.STALLED_BARRIER: Colors.YELLOW + "S" + Colors.RESET,
        WarpState.DONE: Colors.DIM + "D" + Colors.RESET,
    }
    return state_colors.get(state, "?")


def format_bank_state(state: str) -> str:
    """Format bank state with color."""
    if state == "░░":
        return Colors.DIM + "░░" + Colors.RESET
    elif state == "██":
        return Colors.GREEN + "██" + Colors.RESET
    elif state == "▓▓":
        return Colors.BLUE + "▓▓" + Colors.RESET
    elif state == "XX":
        return Colors.RED + Colors.BOLD + "XX" + Colors.RESET
    return state


def format_collector_slot(char: str) -> str:
    """Format collector slot with color."""
    if char == "■":
        return Colors.GREEN + "■" + Colors.RESET
    elif char == "□":
        return Colors.YELLOW + "□" + Colors.RESET
    elif char == "◐":
        return Colors.CYAN + "◐" + Colors.RESET
    else:
        return Colors.DIM + "·" + Colors.RESET


def render_gemm_matrix(
    tracker: GEMMTracker, max_display: int | None = None, box_width: int | None = None
) -> list[str]:
    """
    Render the GEMM output matrix showing warp ownership and update flashing.

    Shows element state progression:
    - Not started: dim warp ID (W0, W1, etc.)
    - Partial progress: ░░, ▓░ based on FMA progress
    - Computed (all FFMAs done, ST pending): ▓▓ in yellow
    - Stored (truly complete): ██ in green
    - Updating this cycle: bright green flash

    Args:
        tracker: GEMMTracker with current state
        max_display: Maximum rows/cols to display. If None, show full matrix.
        box_width: Box width to use. If None, calculate from content.

    Returns:
        List of lines to display
    """
    lines = []
    M, N, K = tracker.M, tracker.N, tracker.K

    # Display the full matrix - let the box grow to fit
    # If this causes wrapping, user can widen their terminal
    if max_display is None:
        max_display = max(M, N)  # Show full matrix

    display_rows = min(M, max_display)
    display_cols = min(N, max_display)
    truncated_rows = display_rows < M
    truncated_cols = display_cols < N

    # Calculate box width based on actual content if not provided
    # "│ XX: " (6) + cols*3 + optional "·· " (3) + padding + "│" (1)
    if box_width is None:
        content_width = 6 + display_cols * 3 + (3 if truncated_cols else 0)
        box_width = content_width + 2  # Just enough for content

    # Header
    pending_stores = tracker.get_pending_store_count()
    status = (
        "COMPLETE"
        if tracker.is_truly_complete()
        else (f"ST pending: {pending_stores}" if pending_stores > 0 else "computing")
    )
    header = f"OUTPUT MATRIX C[{M}×{N}] = A×B  [{status}]"
    lines.append(draw_box(header, box_width))

    # Column headers (warp IDs for first row of each column group)
    col_header = "│     "
    for col in range(display_cols):
        warp_id = tracker.get_warp_for_element(0, col)
        color = Colors.WARP_COLORS[warp_id % len(Colors.WARP_COLORS)]
        col_header += f"{color}{col:2d}{Colors.RESET} "
    if truncated_cols:
        col_header += "··"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", col_header))
    col_header += " " * max(0, box_width - 1 - clean_len) + "│"
    lines.append(col_header)

    # Matrix rows
    for row in range(display_rows):
        row_str = f"│ {row:2d}: "
        for col in range(display_cols):
            warp_id = tracker.get_warp_for_element(row, col)
            progress = tracker.partial_sums[row][col]
            updated = (row, col) in tracker.updated_this_cycle
            computed = progress >= K
            stored = tracker.is_element_stored(row, col)

            # Choose display character and color
            warp_color = Colors.WARP_COLORS[warp_id % len(Colors.WARP_COLORS)]

            if updated:
                # Flash bright green on FMA update
                row_str += f"{Colors.BG_GREEN}{Colors.BLACK}{warp_id:2d}{Colors.RESET} "
            elif stored:
                # Truly complete (stored to memory) - solid green
                row_str += f"{Colors.GREEN}██{Colors.RESET} "
            elif computed:
                # Computed but not stored - yellow/amber to indicate pending ST
                row_str += f"{Colors.YELLOW}▓▓{Colors.RESET} "
            elif progress > 0:
                # Partial progress - show dots based on progress
                # Use block characters to show fill level
                fill = int((progress / K) * 2)  # 0, 1, or 2 chars filled
                if fill >= 2:
                    row_str += f"{warp_color}▓░{Colors.RESET} "
                elif fill >= 1:
                    row_str += f"{warp_color}░▓{Colors.RESET} "
                else:
                    row_str += f"{warp_color}░░{Colors.RESET} "
            else:
                # Not started - show warp ID dimmed
                row_str += f"{Colors.DIM}{warp_id:2d}{Colors.RESET} "

        if truncated_cols:
            row_str += "··"

        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", row_str))
        row_str += " " * max(0, box_width - 1 - clean_len) + "│"
        lines.append(row_str)

    if truncated_rows:
        lines.append("│ ··" + " " * (box_width - 5) + "│")

    # Progress summary - now shows both computed and stored counts
    total_ops = M * N * K
    completed_ops = sum(sum(row) for row in tracker.partial_sums)
    pct = (completed_ops / total_ops * 100) if total_ops > 0 else 0
    elements_computed = sum(
        1 for r in range(M) for c in range(N) if tracker.partial_sums[r][c] >= K
    )
    elements_stored = sum(1 for r in range(M) for c in range(N) if tracker.stored[r][c])

    summary = f"│ FMAs: {completed_ops}/{total_ops} ({pct:.1f}%)  "
    summary += f"Computed: {elements_computed}/{M * N}  "
    summary += f"{Colors.GREEN}Stored: {elements_stored}/{M * N}{Colors.RESET}"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", summary))
    summary += " " * max(0, box_width - 1 - clean_len) + "│"
    lines.append(summary)

    # Legend - updated to show computed vs stored distinction
    legend = "│ "
    legend += f"{Colors.DIM}##=idle(warp#){Colors.RESET} "
    legend += "░=partial "
    legend += f"{Colors.YELLOW}▓▓=computed{Colors.RESET} "
    legend += f"{Colors.GREEN}██=stored{Colors.RESET} "
    legend += f"{Colors.BG_GREEN}{Colors.BLACK}##=updating{Colors.RESET}"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", legend))
    legend += " " * max(0, box_width - 1 - clean_len) + "│"
    lines.append(legend)

    lines.append(draw_box_end(box_width))
    return lines


def visualize_sm_state(
    sm: SMSim, cycle_status: dict, gemm_tracker: GEMMTracker | None = None
) -> list[str]:
    """
    Generate visualization lines for current SM state.

    Returns:
        List of lines to display.
    """
    lines = []
    vis = sm.get_visualization()
    config = sm.config

    # Calculate box width based on matrix size (if tracking GEMM)
    # Each matrix element: 3 chars (2 display + 1 space)
    # Overhead: "│ XX: " (6) + " │" (2) = 8
    if gemm_tracker is not None:
        matrix_width = 6 + max(gemm_tracker.M, gemm_tracker.N) * 3 + 2
        box_width = max(matrix_width, 75)  # At least 75 for standard content
    else:
        box_width = 75
    content_width = box_width - 1  # For padding calculations

    # Header with cycle and energy
    energy_str = format_energy(vis["energy_pj"])
    header = (
        f"Cycle {vis['cycle']:4d} │ "
        f"State: {vis['state']:8s} │ "
        f"Instructions: {vis['total_instructions']:4d} │ "
        f"Bank Conflicts: {vis['total_bank_conflicts']:3d} │ "
        f"Energy: {energy_str}"
    )
    lines.append(Colors.BOLD + header + Colors.RESET)
    lines.append("═" * box_width)
    lines.append("")

    # Warp Schedulers section
    warp_header = (
        f"WARP SCHEDULERS ({config.num_partitions} partitions × "
        f"{config.max_warps_per_partition} warps)"
    )
    lines.append(draw_box(warp_header, box_width))

    for p_idx, p_vis in enumerate(vis["partitions"]):
        warp_line = f"│ P{p_idx}: "
        for char in p_vis["warp_states"]:
            state = WarpState[char] if char in WarpState.__members__ else WarpState.INACTIVE
            # Map single-char back to state
            char_to_state = {
                "·": WarpState.INACTIVE,
                "R": WarpState.READY,
                "I": WarpState.ISSUED,
                "E": WarpState.EXECUTING,
                "W": WarpState.STALLED_RAW,
                "B": WarpState.STALLED_BANK,
                "M": WarpState.STALLED_MEM,
                "S": WarpState.STALLED_BARRIER,
                "D": WarpState.DONE,
            }
            state = char_to_state.get(char, WarpState.INACTIVE)
            warp_line += format_warp_state(state)
        warp_line += "  "

        # Show issued instruction if any
        if cycle_status and p_idx < len(cycle_status.get("partitions", [])):
            p_status = cycle_status["partitions"][p_idx]
            if p_status.get("issued"):
                warp_id, instr = p_status["issued"]
                warp_line += f"│ W{warp_id}: {instr}"

        # Pad to width
        # Remove ANSI codes for length calculation
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", warp_line))
        padding = content_width - clean_len
        warp_line += " " * max(0, padding) + "│"
        lines.append(warp_line)

    lines.append(draw_box_end(box_width))
    lines.append("")

    # Register Files section
    total_regs_k = config.total_registers // 1024
    reg_header = (
        f"REGISTER FILES ({total_regs_k}K Registers, "
        f"{config.register_banks_per_partition} Banks per Partition)"
    )
    lines.append(draw_box(reg_header, box_width))

    for p_idx, p_vis in enumerate(vis["partitions"]):
        bank_line = f"│ P{p_idx}: "
        for state in p_vis["register_banks"][:8]:
            bank_line += f"[{format_bank_state(state)}]"
        bank_line += "..."

        # Show pending writes (memory data trickling back) or conflicts
        pending_writes = sm.partitions[p_idx].register_file.get_pending_write_count()
        if pending_writes > 0:
            bank_line += f"  {Colors.YELLOW}Pending: {pending_writes}/32{Colors.RESET}"
        elif cycle_status and p_idx < len(cycle_status.get("partitions", [])):
            p_status = cycle_status["partitions"][p_idx]
            conflicts = p_status.get("bank_conflicts", [])
            if conflicts:
                bank_line += f"  {Colors.RED}Conflicts: B{conflicts[0]}{Colors.RESET}"

        # Pad to width
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", bank_line))
        padding = content_width - clean_len
        bank_line += " " * max(0, padding) + "│"
        lines.append(bank_line)

    lines.append(draw_box_end(box_width))
    lines.append("")

    # Operand Collectors section - show per-thread operand collection progress
    # Each warp has 32 threads, each thread needs up to 3 source operands
    # Total: 32 threads × 3 sources = 96 operand slots per collector
    coll_header = "OPERAND COLLECTORS (32 threads × 3 sources = 96 slots each)"
    lines.append(draw_box(coll_header, box_width))

    for p_idx, _p_vis in enumerate(vis["partitions"]):
        collector_entries = sm.partitions[p_idx].operand_collector.collectors
        collector_status = sm.partitions[p_idx].operand_collector.get_status()

        # Show detailed progress for each collector
        collector_line = f"│ P{p_idx}: "
        for i, entry in enumerate(collector_entries):
            status = collector_status[i]
            if entry.state == CollectorState.EMPTY:
                collector_line += Colors.DIM + "[----]" + Colors.RESET + " "
            elif entry.state == CollectorState.READY:
                collector_line += Colors.GREEN + f"[W{entry.warp_id}:FIRE]" + Colors.RESET + " "
            else:
                # Show warp ID and progress (e.g., "W0:32/64" or "W0:16+8/48")
                progress = status
                collector_line += (
                    Colors.YELLOW + f"[W{entry.warp_id}:{progress}]" + Colors.RESET + " "
                )

        # Pad to width
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", collector_line))
        padding = content_width - clean_len
        collector_line += " " * max(0, padding) + "│"
        lines.append(collector_line)

    lines.append(draw_box_end(box_width))
    lines.append("")

    # Execution Pipeline section - show ALUs per partition with utilization
    alu_header = (
        f"ALU CLUSTER ({config.cores_per_partition} ALUs × "
        f"{config.pipeline_stages} stages per partition)"
    )
    lines.append(draw_box(alu_header, box_width))

    for p_idx, p_vis in enumerate(vis["partitions"]):
        alu_detail = p_vis.get("alu_detail", {})
        utilization = alu_detail.get("utilization", 0.0)
        alus = alu_detail.get("alus", [])
        num_alus = len(alus)
        busy_count = sum(1 for alu in alus if alu["busy"])

        # Compact ALU display: one char per ALU (█=busy, ·=idle)
        alu_line = f"│ P{p_idx}: "
        for alu in alus:
            if alu["busy"]:
                alu_line += f"{Colors.GREEN}█{Colors.RESET}"
            else:
                alu_line += f"{Colors.DIM}·{Colors.RESET}"

        # Show busy count and utilization
        alu_line += f" {busy_count:2d}/{num_alus} "
        alu_line += f"Util:{Colors.CYAN}{utilization:3.0f}%{Colors.RESET}"

        # Pad to width
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", alu_line))
        padding = content_width - clean_len
        alu_line += " " * max(0, padding) + "│"
        lines.append(alu_line)

    lines.append(draw_box_end(box_width))
    lines.append("")

    # Memory System section - show SM-level LSU activity and MSHR status
    lines.append(draw_box("MEMORY SYSTEM (MIO Queue → MSHR → Cache → DRAM)", box_width))

    # Get SM-level LSU visualization
    lsu_vis = sm.sm_lsu.get_visualization()

    # Show MIO Queue status
    mio_entries = lsu_vis.get("mio_queue", [])
    mio_line = "│ MIO Queue: "
    if not mio_entries:
        mio_line += f"{Colors.DIM}Empty{Colors.RESET}"
    else:
        req_strs = []
        for entry in mio_entries[:6]:  # Show max 6 entries
            part = entry.get("partition", 0)
            warp = entry.get("warp", 0)
            op = "LD" if entry.get("is_load", True) else "ST"
            req_strs.append(f"{Colors.CYAN}P{part}W{warp}:{op}{Colors.RESET}")
        mio_line += " ".join(req_strs)
        if len(mio_entries) > 6:
            mio_line += f" +{len(mio_entries) - 6}"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", mio_line))
    mio_line += " " * max(0, content_width - clean_len) + "│"
    lines.append(mio_line)

    # Show active MSHRs
    active_mshrs = lsu_vis.get("active_mshrs", [])
    mshr_line = "│ MSHRs: "
    if not active_mshrs:
        mshr_line += f"{Colors.DIM}No active cache line requests{Colors.RESET}"
    else:
        mshr_strs = []
        for mshr in active_mshrs[:4]:  # Show max 4 MSHRs
            addr = mshr.get("cache_line", 0)
            waiters = mshr.get("num_waiters", 0)
            cycles = mshr.get("cycles_remaining", 0)
            # Color based on wait time
            if cycles > 100:
                color = Colors.RED
            elif cycles > 50:
                color = Colors.YELLOW
            else:
                color = Colors.GREEN
            mshr_strs.append(f"{color}0x{addr:04x}({waiters}w,{cycles}c){Colors.RESET}")
        mshr_line += " ".join(mshr_strs)
        if len(active_mshrs) > 4:
            mshr_line += f" +{len(active_mshrs) - 4}"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", mshr_line))
    mshr_line += " " * max(0, content_width - clean_len) + "│"
    lines.append(mshr_line)

    # Summary line with LSU statistics
    stats = sm.sm_lsu.get_statistics()
    total_loads = stats.get("total_loads", 0)
    total_stores = stats.get("total_stores", 0)
    primary = stats.get("total_primary_misses", 0)
    secondary = stats.get("total_secondary_misses", 0)

    summary = f"│ LD:{total_loads} ST:{total_stores} Primary:{primary} Secondary:{secondary}"
    if primary + secondary > 0:
        hit_rate = secondary / (primary + secondary) * 100
        summary += f" (MSHR hit:{hit_rate:.0f}%)"

    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", summary))
    summary += " " * max(0, content_width - clean_len) + "│"
    lines.append(summary)

    lines.append(draw_box_end(box_width))
    lines.append("")

    # Energy Breakdown section
    lines.append(draw_box("ENERGY BREAKDOWN", box_width))

    total_energy = vis["energy_pj"]
    instr_energy = vis["total_instructions"] * (
        config.instruction_fetch_energy_pj + config.instruction_decode_energy_pj
    )
    sched_energy = vis["total_instructions"] * config.scheduler_energy_pj
    reg_energy = sum(p.register_file.total_energy_pj for p in sm.partitions)
    alu_energy = sum(p.execution_unit.total_energy_pj for p in sm.partitions)

    energy_line = f"│ Instr Fetch/Decode: {format_energy(instr_energy):>10s} │ "
    energy_line += f"Scheduling: {format_energy(sched_energy):>10s} │ "
    energy_line += f"Reg Access: {format_energy(reg_energy):>10s}"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", energy_line))
    padding = content_width - clean_len
    energy_line += " " * max(0, padding) + "│"
    lines.append(energy_line)

    energy_line2 = f"│ ALU/FMA: {format_energy(alu_energy):>10s} │ "
    energy_line2 += f"Total: {Colors.BOLD}{format_energy(total_energy):>10s}{Colors.RESET} │ "

    # Compute efficiency (useful work / total energy)
    # Useful work = ALU energy only
    efficiency = (alu_energy / max(1, total_energy)) * 100
    energy_line2 += f"Efficiency: {efficiency:.1f}%"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", energy_line2))
    padding = content_width - clean_len
    energy_line2 += " " * max(0, padding) + "│"
    lines.append(energy_line2)

    # Instruction mix line - aggregate per-opcode counts from all partitions
    all_counts: dict[str, int] = {}
    for p in sm.partitions:
        for opcode, count in p.instruction_counts.items():
            all_counts[opcode] = all_counts.get(opcode, 0) + count

    if all_counts:
        # Sort by count descending, show top opcodes
        sorted_ops = sorted(all_counts.items(), key=lambda x: -x[1])
        instr_line = "│ Instr Mix: "
        for idx, (opcode, count) in enumerate(sorted_ops):
            if idx >= 5:  # Show max 5 opcodes
                remaining = sum(c for _, c in sorted_ops[5:])
                if remaining > 0:
                    instr_line += f"+{remaining} other "
                break
            # Color code by type
            if opcode in ("FFMA", "FADD", "FMUL", "FSUB"):
                color = Colors.GREEN  # FP ops
            elif opcode in ("LD", "ST"):
                color = Colors.CYAN  # Memory ops
            else:
                color = Colors.YELLOW  # Other ALU
            instr_line += f"{color}{opcode}:{count}{Colors.RESET} "

        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", instr_line))
        instr_line += " " * max(0, content_width - clean_len) + "│"
        lines.append(instr_line)

    lines.append(draw_box_end(box_width))

    # GEMM Matrix visualization (if tracker provided)
    if gemm_tracker is not None:
        lines.append("")
        lines.extend(render_gemm_matrix(gemm_tracker, box_width=box_width))

    return lines


def run_animation(
    sm: SMSim,
    delay_ms: int = 200,
    max_cycles: int = 1000,
    step_mode: bool = False,
    gemm_tracker: GEMMTracker | None = None,
):
    """
    Run animated visualization of SM execution.

    Args:
        sm: Streaming Multiprocessor simulation
        delay_ms: Delay between frames in milliseconds
        max_cycles: Maximum cycles to run
        step_mode: If True, wait for Enter key between each cycle
        gemm_tracker: Optional GEMM tracker for output matrix visualization
    """

    clear_screen()

    print(Colors.BOLD + "NVIDIA-style Streaming Multiprocessor Simulation" + Colors.RESET)
    print("=" * 75)
    print()
    print(
        f"Configuration: {sm.config.num_partitions} partitions × "
        f"{sm.config.cores_per_partition} cores = {sm.config.total_cores} cores"
    )
    print(f"Register File: {sm.config.total_registers} registers ({sm.config.register_file_kb} KB)")
    if gemm_tracker:
        print(
            f"GEMM: C[{gemm_tracker.M}×{gemm_tracker.N}] = "
            f"A[{gemm_tracker.M}×{gemm_tracker.K}] @ B[{gemm_tracker.K}×{gemm_tracker.N}]"
        )
    print()
    print("Legend: R=Ready E=Executing W=RAW-Stall B=Bank-Stall D=Done")
    print("        ██=Read ▓▓=Write XX=Conflict ░░=Idle")
    print()
    if step_mode:
        print("Step mode: Press Enter to advance, 'q' to quit, 'r' to run continuously")
    else:
        print("Press Ctrl+C to stop")
    print()
    if not step_mode:
        time.sleep(1)

    try:
        run_continuous = False
        while not sm.done and sm.cycle < max_cycles:
            clear_screen()

            # Clear GEMM tracker updates from previous cycle
            if gemm_tracker:
                gemm_tracker.clear_updates()

            # Execute one cycle
            cycle_status = sm.step()
            current_cycle = cycle_status.get("cycle", sm.cycle)

            # Log timeline events and update GEMM tracker
            # Helper to compute global warp ID from partition + local warp
            # Round-robin: global_warp = local_warp * num_partitions + partition
            num_partitions = sm.config.num_partitions

            for p_idx, p_status in enumerate(cycle_status.get("partitions", [])):
                # Log issued instructions (from partition's issued_this_cycle if available)
                issued = p_status.get("issued", None)
                if issued:
                    local_warp_id, instr = issued
                    global_warp_id = local_warp_id * num_partitions + p_idx
                    if timeline_logger.enabled:
                        timeline_logger.log_issue(
                            current_cycle,
                            p_idx,
                            local_warp_id,
                            instr.opcode,
                            instr.dst,
                            instr.src1,
                            instr.src2,
                            instr.latency,
                            "",
                        )
                    # Track store issues for GEMM (use global warp ID)
                    if gemm_tracker and instr.opcode == "ST":
                        gemm_tracker.record_store_issued(global_warp_id)

                # Log ALU completions
                completed = p_status.get("completed", [])
                for local_warp_id, dst_reg, _results in completed:
                    global_warp_id = local_warp_id * num_partitions + p_idx
                    if timeline_logger.enabled:
                        timeline_logger.log_complete(
                            current_cycle, p_idx, local_warp_id, "ALU", dst_reg, ""
                        )
                    # Update GEMM tracker (use global warp ID)
                    if gemm_tracker:
                        gemm_tracker.record_fma_completion(global_warp_id)

                # Log memory completions
                mem_completed = p_status.get("memory_completed", [])
                for local_warp_id, dst_reg, _data in mem_completed:
                    global_warp_id = local_warp_id * num_partitions + p_idx
                    if timeline_logger.enabled:
                        op_type = "LD" if dst_reg >= 0 else "ST"
                        timeline_logger.log_complete(
                            current_cycle, p_idx, local_warp_id, op_type, dst_reg, ""
                        )
                    # Track store completions for GEMM (use global warp ID)
                    if gemm_tracker and dst_reg == -1:
                        gemm_tracker.record_store_complete(global_warp_id)

            # Generate and display visualization
            lines = visualize_sm_state(sm, cycle_status, gemm_tracker)
            for line in lines:
                print(line)

            if step_mode and not run_continuous:
                print()
                print(
                    Colors.DIM + "[Enter]=step  [r]=run  [q]=quit" + Colors.RESET,
                    end="",
                    flush=True,
                )
                try:
                    user_input = input().strip().lower()
                    if user_input == "q":
                        break
                    elif user_input == "r":
                        run_continuous = True
                except EOFError:
                    break
            elif delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user.")

    # Final statistics
    print()
    print(Colors.BOLD + "═" * 75 + Colors.RESET)
    print(Colors.BOLD + "FINAL STATISTICS" + Colors.RESET)
    print("═" * 75)

    stats = sm.get_statistics()
    print(f"Total Cycles:        {stats['cycles']}")
    print(f"Total Instructions:  {stats['instructions']}")
    print(f"IPC:                 {stats['ipc']:.2f}")
    print(f"Total Stalls:        {stats['stalls']}")
    print(f"Bank Conflicts:      {stats['bank_conflicts']}")
    print(f"Total Energy:        {format_energy(stats['energy_pj'])}")

    # Per-partition stats
    print()
    print("Per-Partition Statistics:")
    for i, p_stats in enumerate(stats["partition_stats"]):
        print(
            f"  P{i}: {p_stats['instructions']} instrs, "
            f"{p_stats['stalls']} stalls, "
            f"{format_energy(p_stats['energy_pj'])}"
        )

    # Per-opcode instruction counts
    print()
    print("Instruction Mix:")
    for i, p_stats in enumerate(stats["partition_stats"]):
        counts = p_stats.get("instruction_counts", {})
        if counts:
            count_str = " ".join(
                f"{op}:{cnt}" for op, cnt in sorted(counts.items(), key=lambda x: -x[1])
            )
            print(f"  P{i}: {count_str}")

    # Timeline summary
    if timeline_logger.enabled:
        timeline_logger.print_summary()


def print_gemm_tile_mapping(M: int, N: int, num_warps: int, warp_size: int = 32):
    """Print a visual representation of how warps map to output tiles."""
    print(f"\nGEMM Tile Mapping: C[{M}×{N}] = A[{M}×K] @ B[K×{N}]")
    print(f"Total output elements: {M * N}")
    print(f"Elements per warp: {warp_size} (one per thread)")
    print(f"Warps needed: {num_warps}")
    print()

    # Create a grid showing which warp computes which elements
    print("Output matrix C with warp assignments:")
    print("  ", end="")
    for col in range(N):
        print(f"{col:3}", end="")
    print()

    for row in range(M):
        print(f"{row:2} ", end="")
        for col in range(N):
            idx = row * N + col
            warp_id = idx // warp_size
            print(f" W{warp_id}", end="")
        print()
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Animated SIMT Streaming Multiprocessor Visualization"
    )
    parser.add_argument(
        "--warps",
        type=int,
        default=None,
        help="Number of warps (auto-calculated for tiled GEMM)",
    )
    parser.add_argument(
        "--instructions",
        type=int,
        default=16,
        help="Number of instructions per warp (default: 16)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=200,
        help="Delay between frames in milliseconds (default: 200)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode (no animation delay)",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step mode (press Enter to advance each cycle)",
    )
    parser.add_argument(
        "--gemm",
        action="store_true",
        help="Use GEMM workload (sequential, same program in all warps)",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Use tiled GEMM (parallel, each warp computes different output tile)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=8,
        help="M dimension for GEMM output rows (default: 8)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="N dimension for GEMM output cols (default: 8)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="K dimension for GEMM reduction (default: 4)",
    )
    parser.add_argument(
        "--fast-mem",
        action="store_true",
        help="Use fast memory latencies for demo (1 cycle instead of 200)",
    )
    parser.add_argument(
        "--mem-latency",
        type=int,
        default=None,
        help="Memory latency in cycles (default: 200, overrides --fast-mem)",
    )
    parser.add_argument(
        "--warps-per-partition",
        type=int,
        default=None,
        help="Max warps per partition (default: 8, use more to hide memory latency)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=1000,
        help="Maximum cycles to run before stopping (default: 1000)",
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        metavar="FILE",
        help="Write timeline log to FILE",
    )
    parser.add_argument(
        "--timeline-format",
        type=str,
        default="chrome",
        choices=["csv", "chrome"],
        help="Timeline format: csv (spreadsheet) or chrome (Chrome Trace JSON, default)",
    )

    args = parser.parse_args()

    # Create SM with optional config overrides
    config = SIMTConfig()

    # Memory latency configuration
    if args.mem_latency is not None:
        config.l1_miss_latency = args.mem_latency
        print(f"Using memory latency: {args.mem_latency} cycles")
    elif args.fast_mem:
        config.l1_miss_latency = 1
        config.shared_mem_latency = 1
        print("Using fast memory latencies for demo (1 cycle)")

    # Warps per partition configuration
    if args.warps_per_partition is not None:
        config.max_warps_per_partition = args.warps_per_partition
        config.max_warps_per_sm = config.num_partitions * args.warps_per_partition
        print(
            f"Using {args.warps_per_partition} warps per partition ({config.max_warps_per_sm} total)"
        )

    sm = SMSim(config)

    # Enable timeline logging if requested
    if args.timeline:
        timeline_logger.enable(args.timeline, fmt=args.timeline_format)
        print(f"Timeline logging enabled: {args.timeline} (format: {args.timeline_format})")

    # GEMM tracker for visualization (only for tiled mode)
    gemm_tracker = None

    # Determine workload and warp count
    if args.tiled:
        # Tiled GEMM: each warp computes different output elements
        M, N, K = args.m, args.n, args.k
        num_warps = get_gemm_warp_count(M, N)

        # Calculate max warps across all partitions
        warps_per_partition = config.max_warps_per_partition
        num_partitions = config.num_partitions
        max_total_warps = warps_per_partition * num_partitions

        if num_warps > max_total_warps:
            print(
                f"Warning: {M}×{N} output needs {num_warps} warps, "
                f"limiting to {max_total_warps} ({num_partitions} partitions × "
                f"{warps_per_partition} warps)"
            )
            num_warps = max_total_warps

        # Create GEMM tracker for visualization
        gemm_tracker = GEMMTracker(M, N, K, config.warp_size)

        print_gemm_tile_mapping(M, N, num_warps)

        # Distribute warps across partitions using round-robin (NVIDIA-style)
        # warp_id 0 → P0, 1 → P1, 2 → P2, 3 → P3, 4 → P0, 5 → P1, ...
        partition_warp_counts = [0] * num_partitions

        for global_warp_id in range(num_warps):
            # Round-robin distribution across partitions
            partition_id = global_warp_id % num_partitions
            local_warp_id = global_warp_id // num_partitions

            program = create_tiled_gemm_program(M, N, K, global_warp_id)
            sm.partitions[partition_id].load_program(local_warp_id, program)
            partition_warp_counts[partition_id] += 1

            tile_info = get_warp_tile_info(global_warp_id, M, N)
            print(
                f"  P{partition_id}:W{local_warp_id} (global {global_warp_id}): "
                f"rows {tile_info[0]}-{tile_info[2]}, cols {tile_info[1]}-{tile_info[3]} "
                f"({len(program)} instrs)"
            )

        # Activate warps in each partition that has work
        for p_id, warp_count in enumerate(partition_warp_counts):
            if warp_count > 0:
                sm.partitions[p_id].activate_warps(warp_count)

        # Set SM state to EXECUTE
        sm.state = SMState.EXECUTE
        partitions_used = sum(1 for c in partition_warp_counts if c > 0)
        total_cores = partitions_used * config.cores_per_partition
        print(
            f"\nRunning tiled GEMM: C[{M}×{N}] = A[{M}×{K}] @ B[{K}×{N}]"
            f"\n  Using {partitions_used} partitions, {num_warps} warps, "
            f"{total_cores} ALU cores"
        )

    elif args.gemm:
        # Original sequential GEMM (same program in all warps)
        num_warps = args.warps if args.warps else 4
        program = create_gemm_program(16, 16, args.k)
        print(f"Running sequential GEMM with K={args.k} ({len(program)} instructions per warp)")
        sm.load_uniform_program(program, num_warps)
        sm.activate_warps(num_warps)

    else:
        # Test program
        num_warps = args.warps if args.warps else 4
        program = create_test_program(args.instructions)
        print(f"Running test program ({len(program)} instructions per warp)")
        sm.load_uniform_program(program, num_warps)
        sm.activate_warps(num_warps)

    # Run animation
    delay = 0 if args.fast else args.delay
    run_animation(
        sm,
        delay_ms=delay,
        max_cycles=args.max_cycles,
        step_mode=args.step,
        gemm_tracker=gemm_tracker,
    )


if __name__ == "__main__":
    main()
