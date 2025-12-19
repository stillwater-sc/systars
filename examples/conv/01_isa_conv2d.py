#!/usr/bin/env python3
"""
ISA-Level Conv2D Instruction Demo.

This example demonstrates how to use the Conv2D ISA instruction, which
automatically handles tiled 2D convolution for feature maps larger than
the systolic array.

The Conv2D instruction:
1. Accepts high-level configuration (dimensions, addresses, kernel params)
2. Automatically selects optimal dataflow (OS, WS)
3. Maps convolution to matrix multiplication (im2col style)
4. Tiles the computation to fit the array
5. Generates internal command sequences (LOAD, PRELOAD, COMPUTE, STORE)
6. Manages double buffering for latency hiding

Convolution is mapped to matrix multiplication:
    M = batch * out_height * out_width (output spatial positions)
    N = channels_out (output channels / filters)
    K = kernel_h * kernel_w * channels_in (flattened kernel)

Usage:
    python 01_isa_conv2d.py [options]

Examples:
    # Simple 8x8 input, 3x3 kernel, 16 output channels
    python 01_isa_conv2d.py --in-h 8 --in-w 8 --in-c 3 --out-c 16 --kernel 3

    # Larger input with more channels
    python 01_isa_conv2d.py --in-h 32 --in-w 32 --in-c 64 --out-c 128 --kernel 3
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from amaranth.sim import Simulator, Tick

from systars.config import SystolicConfig
from systars.isa.conv2d import Conv2d, Conv2dCmd, InternalOpcode


def opcode_name(opcode: int) -> str:
    """Convert opcode to human-readable name."""
    names = {
        InternalOpcode.EXEC_CONFIG: "CONFIG",
        InternalOpcode.EXEC_PRELOAD: "PRELOAD",
        InternalOpcode.EXEC_COMPUTE: "COMPUTE",
        InternalOpcode.LOAD_X: "LOAD_X",
        InternalOpcode.LOAD_F: "LOAD_F",
        InternalOpcode.LOAD_B: "LOAD_B",
        InternalOpcode.STORE_Y: "STORE_Y",
    }
    return names.get(opcode, f"UNKNOWN({opcode})")


def compute_output_dims(
    in_h: int,
    in_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> tuple[int, int]:
    """Compute output dimensions for convolution."""
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    return out_h, out_w


def run_conv2d_demo(
    batch: int,
    in_h: int,
    in_w: int,
    in_c: int,
    out_c: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    array_dim: int,
    use_bias: bool,
    verbose: bool = True,
):
    """
    Run the Conv2D ISA instruction demonstration.

    Args:
        batch: Batch size
        in_h: Input height
        in_w: Input width
        in_c: Input channels
        out_c: Output channels (number of filters)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride_h: Stride in height dimension
        stride_w: Stride in width dimension
        pad_h: Padding in height dimension
        pad_w: Padding in width dimension
        array_dim: Systolic array dimension (assumed square)
        use_bias: Whether to use bias
        verbose: Print detailed output
    """
    print("=" * 70)
    print("ISA-Level Conv2D Instruction Demo")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Problem Setup
    # -------------------------------------------------------------------------
    print("\n1. Problem Setup")
    print("-" * 40)

    # Compute output dimensions
    out_h, out_w = compute_output_dims(
        in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
    )

    print(f"   Input:  X[{batch}, {in_h}, {in_w}, {in_c}]  (NHWC format)")
    print(f"   Filter: F[{kernel_h}, {kernel_w}, {in_c}, {out_c}]")
    print(f"   Output: Y[{batch}, {out_h}, {out_w}, {out_c}]")
    if use_bias:
        print(f"   Bias:   B[{out_c}]")
    print(f"\n   Stride: ({stride_h}, {stride_w})")
    print(f"   Padding: ({pad_h}, {pad_w})")

    # Matrix multiplication mapping
    M = batch * out_h * out_w  # Output spatial positions
    N = out_c  # Output channels
    K = kernel_h * kernel_w * in_c  # Flattened kernel

    print("\n   Matrix multiplication mapping:")
    print(f"   M = batch * out_h * out_w = {batch} * {out_h} * {out_w} = {M}")
    print(f"   N = out_c = {N}")
    print(f"   K = kernel_h * kernel_w * in_c = {kernel_h} * {kernel_w} * {in_c} = {K}")

    # Calculate tiling
    tiles_M = (M + array_dim - 1) // array_dim
    tiles_N = (N + array_dim - 1) // array_dim
    tiles_K = (K + array_dim - 1) // array_dim
    total_output_tiles = tiles_M * tiles_N
    total_k_iterations = total_output_tiles * tiles_K

    print(f"\n   Array dimension: {array_dim}x{array_dim}")
    print(f"   M tiles: {tiles_M}, N tiles: {tiles_N}, K tiles: {tiles_K}")
    print(f"   Total output tiles: {total_output_tiles}")
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

    # Create Conv2D instruction instance
    conv2d = Conv2d(config)

    # -------------------------------------------------------------------------
    # 3. Memory Layout
    # -------------------------------------------------------------------------
    print("\n3. Memory Layout")
    print("-" * 40)

    # DRAM addresses
    X_ADDR = 0x1000_0000
    F_ADDR = 0x2000_0000
    Y_ADDR = 0x3000_0000
    B_ADDR = 0x4000_0000 if use_bias else 0x0

    # Calculate sizes
    input_bytes = config.input_bits // 8
    acc_bytes = config.acc_bits // 8

    x_size = batch * in_h * in_w * in_c * input_bytes
    f_size = kernel_h * kernel_w * in_c * out_c * input_bytes
    y_size = batch * out_h * out_w * out_c * acc_bytes
    b_size = out_c * acc_bytes if use_bias else 0

    print(f"   X: 0x{X_ADDR:08X} ({x_size} bytes)")
    print(f"   F: 0x{F_ADDR:08X} ({f_size} bytes)")
    print(f"   Y: 0x{Y_ADDR:08X} ({y_size} bytes)")
    if use_bias:
        print(f"   B: 0x{B_ADDR:08X} ({b_size} bytes)")

    # -------------------------------------------------------------------------
    # 4. Simulation
    # -------------------------------------------------------------------------
    print("\n4. Running Amaranth Simulation")
    print("-" * 40)

    commands_emitted = []
    dataflow_selected = None
    final_M = None
    final_N = None
    final_K = None

    def testbench():
        nonlocal dataflow_selected, final_M, final_N, final_K

        # Always ready to accept commands
        yield conv2d.cmd_ready.eq(1)

        # --- Configure input dimensions ---
        print("   Configuring input dimensions...")
        yield conv2d.cfg_valid.eq(1)
        yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_INPUT_DIMS)
        yield conv2d.cfg_data.eq((in_c << 40) | (in_w << 24) | (in_h << 8) | batch)
        yield Tick()
        yield conv2d.cfg_valid.eq(0)
        yield Tick()

        # --- Configure output dimensions ---
        print("   Configuring output dimensions...")
        yield conv2d.cfg_valid.eq(1)
        yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_OUTPUT_DIMS)
        yield conv2d.cfg_data.eq((out_c << 32) | (out_w << 16) | out_h)
        yield Tick()
        yield conv2d.cfg_valid.eq(0)
        yield Tick()

        # --- Configure kernel dimensions ---
        print("   Configuring kernel dimensions...")
        yield conv2d.cfg_valid.eq(1)
        yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_KERNEL_DIMS)
        yield conv2d.cfg_data.eq((kernel_w << 8) | kernel_h)
        yield Tick()
        yield conv2d.cfg_valid.eq(0)
        yield Tick()

        # --- Configure stride ---
        if stride_h != 1 or stride_w != 1:
            print("   Configuring stride...")
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_STRIDE)
            yield conv2d.cfg_data.eq((stride_w << 8) | stride_h)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

        # --- Configure padding ---
        if pad_h != 0 or pad_w != 0:
            print("   Configuring padding...")
            yield conv2d.cfg_valid.eq(1)
            yield conv2d.cfg_cmd.eq(Conv2dCmd.CONFIG_PADDING)
            yield conv2d.cfg_data.eq((pad_w << 8) | pad_h)
            yield Tick()
            yield conv2d.cfg_valid.eq(0)
            yield Tick()

        # --- Configure addresses ---
        print("   Configuring addresses...")
        for _name, cmd, addr in [
            ("X", Conv2dCmd.CONFIG_X_ADDR, X_ADDR),
            ("F", Conv2dCmd.CONFIG_F_ADDR, F_ADDR),
            ("Y", Conv2dCmd.CONFIG_Y_ADDR, Y_ADDR),
            ("B", Conv2dCmd.CONFIG_B_ADDR, B_ADDR),
        ]:
            if addr != 0:
                yield conv2d.cfg_valid.eq(1)
                yield conv2d.cfg_cmd.eq(cmd)
                yield conv2d.cfg_data.eq(addr)
                yield Tick()
                yield conv2d.cfg_valid.eq(0)
                yield Tick()

        # --- Start execution ---
        print("   Starting execution...")
        yield conv2d.cfg_valid.eq(1)
        yield conv2d.cfg_cmd.eq(Conv2dCmd.START)
        yield Tick()
        yield conv2d.cfg_valid.eq(0)

        # --- Monitor execution ---
        print("   Monitoring command generation...")
        cycle = 0
        max_cycles = 2000  # Safety limit

        while cycle < max_cycles:
            yield Tick()
            cycle += 1

            # Check for commands
            valid = yield conv2d.cmd_valid
            if valid:
                opcode = yield conv2d.cmd_opcode
                rs1 = yield conv2d.cmd_rs1
                rs2 = yield conv2d.cmd_rs2
                rd = yield conv2d.cmd_rd

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
            done = yield conv2d.done
            if done:
                dataflow_selected = yield conv2d.selected_dataflow
                final_M = yield conv2d.param_M
                final_N = yield conv2d.param_N
                final_K = yield conv2d.param_K
                print(f"   Completed in {cycle} cycles")
                break

        if cycle >= max_cycles:
            print(f"   WARNING: Hit cycle limit ({max_cycles})")

    # Run simulation
    sim = Simulator(conv2d)
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
    dataflow_names = {0: "Output-Stationary", 1: "Weight-Stationary"}
    print(f"\n   Dataflow selected: {dataflow_names.get(dataflow_selected, 'Unknown')}")

    # Verify M, N, K calculation
    print("\n   Dimension verification:")
    print(f"      M: expected={M}, actual={final_M} {'OK' if final_M == M else 'MISMATCH'}")
    print(f"      N: expected={N}, actual={final_N} {'OK' if final_N == N else 'MISMATCH'}")
    print(f"      K: expected={K}, actual={final_K} {'OK' if final_K == K else 'MISMATCH'}")

    # Verify expected counts
    expected_stores = total_output_tiles
    expected_load_x = total_k_iterations
    expected_load_f = total_k_iterations
    expected_load_b = total_output_tiles if use_bias else 0
    expected_preload = total_output_tiles
    expected_compute = total_k_iterations

    print("\n   Expected vs Actual:")
    checks = [
        ("STORE_Y", expected_stores, cmd_counts.get("STORE_Y", 0)),
        ("LOAD_X", expected_load_x, cmd_counts.get("LOAD_X", 0)),
        ("LOAD_F", expected_load_f, cmd_counts.get("LOAD_F", 0)),
        ("LOAD_B", expected_load_b, cmd_counts.get("LOAD_B", 0)),
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
    # 6. Reference Computation (optional)
    # -------------------------------------------------------------------------
    print("\n6. Reference Computation")
    print("-" * 40)

    # Create random test data
    np.random.seed(42)
    X = np.random.randint(-128, 127, (batch, in_h, in_w, in_c), dtype=np.int8)
    F = np.random.randint(-128, 127, (kernel_h, kernel_w, in_c, out_c), dtype=np.int8)

    # Compute reference output using NumPy
    Y_ref = np.zeros((batch, out_h, out_w, out_c), dtype=np.int32)
    for b in range(batch):
        for oh in range(out_h):
            for ow in range(out_w):
                for oc in range(out_c):
                    acc = 0
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            ih = oh * stride_h - pad_h + kh
                            iw = ow * stride_w - pad_w + kw
                            if 0 <= ih < in_h and 0 <= iw < in_w:
                                for ic in range(in_c):
                                    acc += int(X[b, ih, iw, ic]) * int(F[kh, kw, ic, oc])
                    Y_ref[b, oh, ow, oc] = acc

    print(f"   Reference output shape: {Y_ref.shape}")
    print(f"   Reference output range: [{Y_ref.min()}, {Y_ref.max()}]")
    print(f"   Reference output sum: {Y_ref.sum()}")

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
        description="ISA-Level Conv2D Instruction Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--in-h",
        type=int,
        default=8,
        help="Input height (default: 8)",
    )
    parser.add_argument(
        "--in-w",
        type=int,
        default=8,
        help="Input width (default: 8)",
    )
    parser.add_argument(
        "--in-c",
        type=int,
        default=3,
        help="Input channels (default: 3)",
    )
    parser.add_argument(
        "--out-c",
        type=int,
        default=16,
        help="Output channels (default: 16)",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Kernel size (square, default: 3)",
    )
    parser.add_argument(
        "--kernel-h",
        type=int,
        default=None,
        help="Kernel height (overrides --kernel)",
    )
    parser.add_argument(
        "--kernel-w",
        type=int,
        default=None,
        help="Kernel width (overrides --kernel)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride (square, default: 1)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Padding (same for h and w, default: 0 for valid padding)",
    )
    parser.add_argument(
        "--array-dim",
        type=int,
        default=4,
        help="Systolic array dimension (default: 4)",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Include bias in convolution",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Handle kernel dimensions
    kernel_h = args.kernel_h if args.kernel_h is not None else args.kernel
    kernel_w = args.kernel_w if args.kernel_w is not None else args.kernel

    success = run_conv2d_demo(
        batch=args.batch,
        in_h=args.in_h,
        in_w=args.in_w,
        in_c=args.in_c,
        out_c=args.out_c,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=args.stride,
        stride_w=args.stride,
        pad_h=args.pad,
        pad_w=args.pad,
        array_dim=args.array_dim,
        use_bias=args.bias,
        verbose=not args.quiet,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
