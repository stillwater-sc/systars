#!/usr/bin/env python3
"""
ISA-Level Matmul Instruction Demo.

This example demonstrates how to use the Matmul ISA instruction, which
automatically handles tiled matrix multiplication for matrices larger than
the systolic array.

The Matmul instruction:
1. Accepts high-level configuration (dimensions, addresses, strides)
2. Automatically selects optimal dataflow (OS, AS, BS)
3. Tiles the computation to fit the array
4. Generates internal command sequences (LOAD, PRELOAD, COMPUTE, STORE)
5. Manages double buffering for latency hiding

This is a higher level of abstraction than manual command sequencing -
the user just configures the operation and issues START.

Usage:
    python 07_isa_matmul.py [--M rows] [--N cols] [--K inner] [--array-dim dim]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.isa.matmul import InternalOpcode, Matmul, MatmulCmd


def opcode_name(opcode: int) -> str:
    """Convert opcode to human-readable name."""
    names = {
        InternalOpcode.EXEC_CONFIG: "CONFIG",
        InternalOpcode.EXEC_PRELOAD: "PRELOAD",
        InternalOpcode.EXEC_COMPUTE: "COMPUTE",
        InternalOpcode.LOAD_A: "LOAD_A",
        InternalOpcode.LOAD_B: "LOAD_B",
        InternalOpcode.LOAD_D: "LOAD_D",
        InternalOpcode.STORE_C: "STORE_C",
    }
    return names.get(opcode, f"UNKNOWN({opcode})")


def run_matmul_demo(M: int, N: int, K: int, array_dim: int, verbose: bool = True):
    """
    Run the Matmul ISA instruction demonstration.

    Args:
        M: Number of rows in output matrix C
        N: Number of columns in output matrix C
        K: Inner dimension (cols of A, rows of B)
        array_dim: Systolic array dimension (assumed square)
        verbose: Print detailed output
    """
    print("=" * 70)
    print("ISA-Level Matmul Instruction Demo")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Problem Setup
    # -------------------------------------------------------------------------
    print("\n1. Problem Setup")
    print("-" * 40)
    print(f"   Computing: C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}] + D[{M}x{N}]")

    # Calculate tiling
    tiles_M = (M + array_dim - 1) // array_dim
    tiles_N = (N + array_dim - 1) // array_dim
    tiles_K = (K + array_dim - 1) // array_dim
    total_tiles = tiles_M * tiles_N
    total_k_iterations = total_tiles * tiles_K

    print(f"\n   Array dimension: {array_dim}x{array_dim}")
    print(f"   Output tiles: {tiles_M} x {tiles_N} = {total_tiles} tiles")
    print(f"   K iterations per tile: {tiles_K}")
    print(f"   Total K iterations: {total_k_iterations}")

    # -------------------------------------------------------------------------
    # 2. Hardware Configuration
    # -------------------------------------------------------------------------
    print("\n2. Hardware Configuration")
    print("-" * 40)

    config = SystolicConfig(
        grid_rows=array_dim,
        grid_cols=array_dim,
        tile_rows=1,
        tile_cols=1,
        input_bits=8,
        acc_bits=32,
    )

    print(f"   Grid: {config.grid_rows}x{config.grid_cols}")
    print(f"   Input bits: {config.input_bits}")
    print(f"   Accumulator bits: {config.acc_bits}")

    # Create Matmul instruction instance
    matmul = Matmul(config)

    # -------------------------------------------------------------------------
    # 3. Memory Layout
    # -------------------------------------------------------------------------
    print("\n3. Memory Layout")
    print("-" * 40)

    # DRAM addresses
    A_ADDR = 0x1000_0000
    B_ADDR = 0x2000_0000
    C_ADDR = 0x3000_0000
    D_ADDR = 0x4000_0000  # Bias

    # Strides (row stride in bytes)
    input_bytes = config.input_bits // 8
    acc_bytes = config.acc_bits // 8
    A_STRIDE = K * input_bytes  # A is MxK
    B_STRIDE = N * input_bytes  # B is KxN
    C_STRIDE = N * acc_bytes  # C is MxN
    D_STRIDE = N * acc_bytes  # D is MxN

    print(f"   A: 0x{A_ADDR:08X} (stride={A_STRIDE} bytes)")
    print(f"   B: 0x{B_ADDR:08X} (stride={B_STRIDE} bytes)")
    print(f"   C: 0x{C_ADDR:08X} (stride={C_STRIDE} bytes)")
    print(f"   D: 0x{D_ADDR:08X} (stride={D_STRIDE} bytes, bias)")

    # -------------------------------------------------------------------------
    # 4. Simulation
    # -------------------------------------------------------------------------
    print("\n4. Running Amaranth Simulation")
    print("-" * 40)

    commands_emitted = []
    dataflow_selected = None
    final_progress = None

    def testbench():
        nonlocal dataflow_selected, final_progress

        # Always ready to accept commands
        yield matmul.cmd_ready.eq(1)

        # --- Configure dimensions ---
        print("   Configuring dimensions...")
        yield matmul.cfg_valid.eq(1)
        yield matmul.cfg_cmd.eq(MatmulCmd.CONFIG_DIMS)
        yield matmul.cfg_data.eq((K << 32) | (N << 16) | M)
        yield Tick()
        yield matmul.cfg_valid.eq(0)
        yield Tick()

        # --- Configure addresses ---
        print("   Configuring addresses...")
        for _name, cmd, addr in [
            ("A", MatmulCmd.CONFIG_A_ADDR, A_ADDR),
            ("B", MatmulCmd.CONFIG_B_ADDR, B_ADDR),
            ("C", MatmulCmd.CONFIG_C_ADDR, C_ADDR),
            ("D", MatmulCmd.CONFIG_D_ADDR, D_ADDR),
        ]:
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(cmd)
            yield matmul.cfg_data.eq(addr)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

        # --- Configure strides ---
        print("   Configuring strides...")
        for _name, cmd, stride in [
            ("A", MatmulCmd.CONFIG_A_STRIDE, A_STRIDE),
            ("B", MatmulCmd.CONFIG_B_STRIDE, B_STRIDE),
            ("C", MatmulCmd.CONFIG_C_STRIDE, C_STRIDE),
            ("D", MatmulCmd.CONFIG_D_STRIDE, D_STRIDE),
        ]:
            yield matmul.cfg_valid.eq(1)
            yield matmul.cfg_cmd.eq(cmd)
            yield matmul.cfg_data.eq(stride)
            yield Tick()
            yield matmul.cfg_valid.eq(0)
            yield Tick()

        # --- Start execution ---
        print("   Starting execution...")
        yield matmul.cfg_valid.eq(1)
        yield matmul.cfg_cmd.eq(MatmulCmd.START)
        yield Tick()
        yield matmul.cfg_valid.eq(0)

        # --- Monitor execution ---
        print("   Monitoring command generation...")
        cycle = 0
        max_cycles = 1000  # Safety limit

        while cycle < max_cycles:
            yield Tick()
            cycle += 1

            # Check for commands
            valid = yield matmul.cmd_valid
            if valid:
                opcode = yield matmul.cmd_opcode
                rs1 = yield matmul.cmd_rs1
                rs2 = yield matmul.cmd_rs2
                rd = yield matmul.cmd_rd

                # Record command
                commands_emitted.append(
                    {
                        "cycle": cycle,
                        "opcode": opcode,
                        "rs1": rs1,
                        "rs2": rs2,
                        "rd": rd,
                    }
                )

                if verbose:
                    name = opcode_name(opcode)
                    print(f"      [{cycle:3d}] {name}: rs1=0x{rs1:X}, rs2=0x{rs2:X}")

            # Check for completion
            done = yield matmul.done
            if done:
                dataflow_selected = yield matmul.selected_dataflow
                final_progress = (
                    (yield matmul.progress_i),
                    (yield matmul.progress_j),
                    (yield matmul.progress_k),
                )
                print(f"   Completed in {cycle} cycles")
                break

        if cycle >= max_cycles:
            print(f"   WARNING: Hit cycle limit ({max_cycles})")

    # Run simulation
    sim = Simulator(matmul)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.run()

    # -------------------------------------------------------------------------
    # 5. Analysis
    # -------------------------------------------------------------------------
    print("\n5. Command Analysis")
    print("-" * 40)

    # Count commands by type
    cmd_counts = {}
    for cmd in commands_emitted:
        name = opcode_name(cmd["opcode"])
        cmd_counts[name] = cmd_counts.get(name, 0) + 1

    print("   Commands generated:")
    for name, count in sorted(cmd_counts.items()):
        print(f"      {name}: {count}")

    # Dataflow selection
    dataflow_names = {0: "Output-Stationary", 1: "A-Stationary", 2: "B-Stationary"}
    print(f"\n   Dataflow selected: {dataflow_names.get(dataflow_selected, 'Unknown')}")

    # Verify expected counts
    expected_stores = total_tiles
    expected_load_a = total_k_iterations
    expected_load_b = total_k_iterations
    expected_load_d = total_tiles  # One bias load per output tile
    expected_preload = total_tiles  # One preload per output tile
    expected_compute = total_k_iterations

    print("\n   Expected vs Actual:")
    checks = [
        ("STORE_C", expected_stores, cmd_counts.get("STORE_C", 0)),
        ("LOAD_A", expected_load_a, cmd_counts.get("LOAD_A", 0)),
        ("LOAD_B", expected_load_b, cmd_counts.get("LOAD_B", 0)),
        ("LOAD_D", expected_load_d, cmd_counts.get("LOAD_D", 0)),
        ("PRELOAD", expected_preload, cmd_counts.get("PRELOAD", 0)),
        ("COMPUTE", expected_compute, cmd_counts.get("COMPUTE", 0)),
    ]

    all_pass = True
    for name, expected, actual in checks:
        status = "OK" if expected == actual else "MISMATCH"
        if expected != actual:
            all_pass = False
        print(f"      {name}: expected={expected}, actual={actual} [{status}]")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_pass:
        print("Demo completed successfully!")
    else:
        print("Demo completed with mismatches - please investigate.")
    print("=" * 70)

    return all_pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ISA-Level Matmul Instruction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--M",
        type=int,
        default=8,
        help="Rows in output matrix C (default: 8)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=8,
        help="Columns in output matrix C (default: 8)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=8,
        help="Inner dimension (default: 8)",
    )
    parser.add_argument(
        "--array-dim",
        type=int,
        default=4,
        help="Systolic array dimension (default: 4)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    success = run_matmul_demo(
        M=args.M,
        N=args.N,
        K=args.K,
        array_dim=args.array_dim,
        verbose=not args.quiet,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
