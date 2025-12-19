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
    --warps N       Number of warps to simulate (default: 4)
    --instructions  Number of instructions per warp (default: 16)
    --delay MS      Delay between frames in milliseconds (default: 200)
    --fast          Fast mode (no animation delay)
    --gemm          Use GEMM workload instead of test program
    --k K           K dimension for GEMM (default: 8)
"""

import argparse
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 3)[0] + "/src")

from systars.simt import (
    SIMTConfig,
    SMSim,
    WarpState,
    create_gemm_program,
    create_test_program,
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


def clear_screen():
    """Clear terminal screen and move cursor to top-left."""
    print("\033[2J\033[H", end="")


def move_cursor(row: int, col: int):
    """Move cursor to specified position."""
    print(f"\033[{row};{col}H", end="")


def draw_box(title: str, width: int = 70) -> str:
    """Draw a box header."""
    padding = width - len(title) - 4
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
    else:
        return Colors.DIM + "·" + Colors.RESET


def visualize_sm_state(sm: SMSim, cycle_status: dict) -> list[str]:
    """
    Generate visualization lines for current SM state.

    Returns:
        List of lines to display.
    """
    lines = []
    vis = sm.get_visualization()
    config = sm.config

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
    lines.append("═" * 75)
    lines.append("")

    # Warp Schedulers section
    lines.append(draw_box("WARP SCHEDULERS (4 partitions × 8 warps)", 75))

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
        import re

        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", warp_line))
        padding = 74 - clean_len
        warp_line += " " * max(0, padding) + "│"
        lines.append(warp_line)

    lines.append(draw_box_end(75))
    lines.append("")

    # Register Files section
    lines.append(draw_box("REGISTER FILES (64K Registers, 16 Banks per Partition)", 75))

    for p_idx, p_vis in enumerate(vis["partitions"]):
        bank_line = f"│ P{p_idx}: "
        for state in p_vis["register_banks"][:8]:
            bank_line += f"[{format_bank_state(state)}]"
        bank_line += "..."

        # Show conflicts
        if cycle_status and p_idx < len(cycle_status.get("partitions", [])):
            p_status = cycle_status["partitions"][p_idx]
            conflicts = p_status.get("bank_conflicts", [])
            if conflicts:
                bank_line += f"  {Colors.RED}Conflicts: B{conflicts[0]}{Colors.RESET}"

        # Pad to width
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", bank_line))
        padding = 74 - clean_len
        bank_line += " " * max(0, padding) + "│"
        lines.append(bank_line)

    lines.append(draw_box_end(75))
    lines.append("")

    # Operand Collectors section
    lines.append(draw_box("OPERAND COLLECTORS (2 per partition)", 75))

    for p_idx, p_vis in enumerate(vis["partitions"]):
        collector_line = f"│ P{p_idx}: "
        for slots, status in zip(p_vis["collectors"], p_vis["collector_status"], strict=True):
            collector_line += "["
            for char in slots:
                collector_line += format_collector_slot(char)
            collector_line += "] "
            if status == "FIRE!":
                collector_line += Colors.GREEN + "FIRE! " + Colors.RESET
            elif status == "empty":
                collector_line += Colors.DIM + "empty " + Colors.RESET
            else:
                collector_line += f"{status}   "

        # Pad to width
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", collector_line))
        padding = 74 - clean_len
        collector_line += " " * max(0, padding) + "│"
        lines.append(collector_line)

    lines.append(draw_box_end(75))
    lines.append("")

    # Execution Pipeline section
    lines.append(draw_box("EXECUTION PIPELINE (4 stages)", 75))

    for p_idx, p_vis in enumerate(vis["partitions"]):
        pipe_line = f"│ P{p_idx}: "
        for i, stage in enumerate(p_vis["pipeline"]):
            stage_names = ["FETCH", "DECODE", "EXEC", "WB"]
            name = stage_names[i] if i < len(stage_names) else f"S{i}"
            if stage != "----":
                pipe_line += f"{Colors.CYAN}{name}:{stage}{Colors.RESET} │ "
            else:
                pipe_line += f"{Colors.DIM}{name}:----{Colors.RESET} │ "

        # Pad to width
        clean_len = len(re.sub(r"\033\[[0-9;]*m", "", pipe_line))
        padding = 74 - clean_len
        pipe_line += " " * max(0, padding) + "│"
        lines.append(pipe_line)

    lines.append(draw_box_end(75))
    lines.append("")

    # Energy Breakdown section
    lines.append(draw_box("ENERGY BREAKDOWN", 75))

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
    padding = 74 - clean_len
    energy_line += " " * max(0, padding) + "│"
    lines.append(energy_line)

    energy_line2 = f"│ ALU/FMA: {format_energy(alu_energy):>10s} │ "
    energy_line2 += f"Total: {Colors.BOLD}{format_energy(total_energy):>10s}{Colors.RESET} │ "

    # Compute efficiency (useful work / total energy)
    # Useful work = ALU energy only
    efficiency = (alu_energy / max(1, total_energy)) * 100
    energy_line2 += f"Efficiency: {efficiency:.1f}%"
    clean_len = len(re.sub(r"\033\[[0-9;]*m", "", energy_line2))
    padding = 74 - clean_len
    energy_line2 += " " * max(0, padding) + "│"
    lines.append(energy_line2)

    lines.append(draw_box_end(75))

    return lines


def run_animation(
    sm: SMSim,
    delay_ms: int = 200,
    max_cycles: int = 1000,
):
    """
    Run animated visualization of SM execution.

    Args:
        sm: Streaming Multiprocessor simulation
        delay_ms: Delay between frames in milliseconds
        max_cycles: Maximum cycles to run
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
    print()
    print("Legend: R=Ready E=Executing W=RAW-Stall B=Bank-Stall D=Done")
    print("        ██=Read ▓▓=Write XX=Conflict ░░=Idle")
    print()
    print("Press Ctrl+C to stop")
    print()
    time.sleep(1)

    try:
        while not sm.done and sm.cycle < max_cycles:
            clear_screen()

            # Execute one cycle
            cycle_status = sm.step()

            # Generate and display visualization
            lines = visualize_sm_state(sm, cycle_status)
            for line in lines:
                print(line)

            if delay_ms > 0:
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Animated SIMT Streaming Multiprocessor Visualization"
    )
    parser.add_argument(
        "--warps",
        type=int,
        default=4,
        help="Number of warps to simulate (default: 4)",
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
        "--gemm",
        action="store_true",
        help="Use GEMM workload instead of test program",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="K dimension for GEMM (default: 8)",
    )

    args = parser.parse_args()

    # Create SM
    config = SIMTConfig()
    sm = SMSim(config)

    # Create program
    if args.gemm:
        program = create_gemm_program(16, 16, args.k)
        print(f"Running GEMM with K={args.k} ({len(program)} instructions per warp)")
    else:
        program = create_test_program(args.instructions)
        print(f"Running test program ({len(program)} instructions per warp)")

    # Load program into warps
    sm.load_uniform_program(program, args.warps)

    # Activate warps
    sm.activate_warps(args.warps)

    # Run animation
    delay = 0 if args.fast else args.delay
    run_animation(sm, delay_ms=delay)


if __name__ == "__main__":
    main()
