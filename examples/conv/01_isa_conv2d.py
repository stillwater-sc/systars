#!/usr/bin/env python3
"""
ISA-Level Conv2D Instruction Demo.

The Conv2D instruction runs a 2D convolution (NHWC) on the systolic array as an
implicit GEMM:

    M = batch * out_h * out_w   (output positions, the rows)
    N = out_c                   (output channels, the columns)
    K = kernel_h * kernel_w * in_c, walked one K tile per kernel offset (kh, kw)
        and chunk of array-dim input channels

For each K tile it LOADs one row per output position (the input pixel that
kernel offset reads, with stride, dilation and padding applied, or a CLEAR of
the row in the padding: zeros, with no L3 read) and the filter tile (one command, corner-turned), then COMPUTEs
output-stationary: the K tiles are one reduction held in the PEs; after the K
tiles it STOREs the output tile (one command). X, F and Y are dense NHWC at any address: the tile's BlockMover moves
rows from and to any byte.

Here a stand-in controller accepts and completes every command, and the demo
prints a summary of the command stream and checks it against the tiling.

Usage:
    python 01_isa_conv2d.py [--batch B] [--in-h H] [--in-w W] [--in-c C] [--out-c C]
                            [--kernel K] [--stride S] [--pad P] [--dilation D]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from amaranth.sim import Simulator

from systars.config import SystolicConfig
from systars.isa.command import CTRL_LOAD, CTRL_STORE
from systars.isa.conv2d import Conv2d, Conv2dCmd, Conv2dError
from systars.util.commands import DmaOpcode, ExecOpcode

X_ADDR, F_ADDR, Y_ADDR = 0x1000_0000, 0x2000_0000, 0x3000_0000


def out_size(size, kernel, stride, pad, dilation):
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def run_conv2d_demo(batch, in_h, in_w, in_c, out_c, kernel, stride, pad, dilation, array_dim):
    out_h = out_size(in_h, kernel, stride, pad, dilation)
    out_w = out_size(in_w, kernel, stride, pad, dilation)
    print("=" * 70)
    print(
        f"ISA Conv2D: X[{batch},{in_h},{in_w},{in_c}] * F[{kernel},{kernel},{in_c},{out_c}]"
        f" -> Y[{batch},{out_h},{out_w},{out_c}]"
    )
    print(f"   stride {stride}, padding {pad}, dilation {dilation}, {array_dim}x{array_dim} array")
    print("=" * 70)

    config = SystolicConfig(
        grid_rows=array_dim, grid_cols=array_dim, sp_capacity_kb=4, acc_capacity_kb=4
    )
    conv = Conv2d(config)
    y_stride = 0  # dense output: out_c results per position
    trace, status = [], {}

    async def controller(ctx):
        """Accept every command as it is pushed; report each STORE done a cycle later."""
        store_done = False
        while True:
            ctx.set(conv.cmd_ready, 1)
            ctx.set(conv.cmd_store_completed, store_done)
            _, _, valid, ctype, opcode, dram = await ctx.tick().sample(
                conv.cmd_valid, conv.cmd_type, conv.cmd_opcode, conv.cmd_dram_addr
            )
            store_done = bool(valid and ctype == CTRL_STORE)
            if valid:
                trace.append((ctype, opcode, dram))

    async def testbench(ctx):
        cmds = [
            (Conv2dCmd.CONFIG_INPUT_DIMS, batch | in_h << 8 | in_w << 24 | in_c << 40),
            (Conv2dCmd.CONFIG_OUTPUT_DIMS, out_h | out_w << 16 | out_c << 32),
            (Conv2dCmd.CONFIG_KERNEL_DIMS, kernel | kernel << 8),
            (Conv2dCmd.CONFIG_STRIDE, stride | stride << 8),
            (Conv2dCmd.CONFIG_PADDING, pad | pad << 8),
            (Conv2dCmd.CONFIG_DILATION, dilation | dilation << 8),
            (Conv2dCmd.CONFIG_X_ADDR, X_ADDR),
            (Conv2dCmd.CONFIG_F_ADDR, F_ADDR),
            (Conv2dCmd.CONFIG_Y_ADDR, Y_ADDR),
            (Conv2dCmd.CONFIG_Y_STRIDE, y_stride),
            (Conv2dCmd.START, 0),
        ]
        for cmd, data in cmds:
            ctx.set(conv.cfg_valid, 1)
            ctx.set(conv.cfg_cmd, cmd)
            ctx.set(conv.cfg_data, data)
            await ctx.tick()
        ctx.set(conv.cfg_valid, 0)
        # Every command completes within a few cycles: bound the wait so a stalled
        # handshake ends the demo instead of hanging it
        m_tiles = -(-(batch * out_h * out_w) // array_dim)
        rows = m_tiles * -(-out_c // array_dim) * kernel * kernel * -(-in_c // array_dim)
        for _ in range(1000 + 64 * rows * (3 * array_dim + 4)):
            _, _, done, error, code = await ctx.tick().sample(
                conv.done, conv.error, conv.error_code
            )
            if done or error:
                status.update(done=done, error=error, code=code)
                return

    sim = Simulator(conv)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.add_process(controller)
    sim.run()

    if not status:
        print("\n   Conv2d did not finish (cycle limit reached)")
        return False
    if status["error"]:
        print(f"\n   Conv2d reported {Conv2dError(status['code']).name}")
        return False

    def tiles(n):
        return [min(array_dim, n - s) for s in range(0, n, array_dim)]

    k_tiles = kernel * kernel * len(tiles(in_c))
    expected = {"loads": 0, "stores": 0, "PRELOAD": 0, "COMPUTE": 0, "COMPUTE_ACCUMULATE": 0}
    for tm in tiles(batch * out_h * out_w):
        for _tn in tiles(out_c):
            for _kh in range(kernel):
                for _kw in range(kernel):
                    for _tk in tiles(in_c):
                        expected["loads"] += tm + 1  # a row per position, the filter tile
            expected["COMPUTE"] += k_tiles  # OS: one per K tile, chained (no PRELOAD)
            expected["stores"] += 1  # the output tile
    actual = {
        "loads": sum(1 for t in trace if t[0] == CTRL_LOAD),
        "stores": sum(1 for t in trace if t[0] == CTRL_STORE),
    }
    for op in ("PRELOAD", "COMPUTE", "COMPUTE_ACCUMULATE"):
        actual[op] = sum(
            1 for t in trace if t[0] not in (CTRL_LOAD, CTRL_STORE) and t[1] == ExecOpcode[op]
        )
    padded = sum(1 for t in trace if t[0] == CTRL_LOAD and t[1] == DmaOpcode.CLEAR)

    print(f"\n   {len(trace)} commands; {padded} input rows are CLEARed (padding)")
    print("   Expected vs actual:")
    ok = True
    for name in expected:
        match = expected[name] == actual[name]
        ok &= match
        print(
            f"      {name:<18} {expected[name]:5d} {actual[name]:5d}  {'OK' if match else 'MISMATCH'}"
        )
    print("\n" + ("Demo completed successfully!" if ok else "Demo completed with mismatches."))
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in-h", type=int, default=6)
    parser.add_argument("--in-w", type=int, default=6)
    parser.add_argument("--in-c", type=int, default=3)
    parser.add_argument("--out-c", type=int, default=8)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--pad", type=int, default=1)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--array-dim", type=int, default=4)
    args = parser.parse_args()
    ok = run_conv2d_demo(
        args.batch, args.in_h, args.in_w, args.in_c, args.out_c, args.kernel,
        args.stride, args.pad, args.dilation, args.array_dim,
    )  # fmt: skip
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
