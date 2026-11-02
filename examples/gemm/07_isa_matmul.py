#!/usr/bin/env python3
"""
ISA-Level Matmul Instruction Demo.

The Matmul instruction takes a high-level description (dimensions, addresses,
strides), tiles the multiply for the array, and issues the commands a host would
issue to the systolic tile: LOAD (one per tile row), CONFIG, PRELOAD, COMPUTE,
STORE (one per tile row). Each command names its target controller and that
controller's own opcode, and is issued only after the previous one completed.

Here a stand-in controller accepts and completes every command, and the demo
prints the command stream and checks its shape against the tiling.

By default the operands use the layout the tile's DMA can move (each row one bus
beat apart, N and K within one tile). ``--any-layout`` uses dense row-major
operands tiled in M, N and K, with Matmul's beat-alignment check turned off.

Usage:
    python 07_isa_matmul.py [--M rows] [--N cols] [--K inner] [--array-dim dim] [--any-layout]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from amaranth.sim import Simulator

from systars.config import SystolicConfig
from systars.isa.command import CTRL_LOAD, CTRL_STORE
from systars.isa.matmul import Matmul, MatmulCmd, MatmulError
from systars.util.commands import ExecOpcode

A_ADDR, B_ADDR, C_ADDR = 0x1000_0000, 0x2000_0000, 0x3000_0000


def command_name(ctype: int, opcode: int) -> str:
    if ctype == CTRL_LOAD:
        return "LOAD"
    if ctype == CTRL_STORE:
        return "STORE"
    return ExecOpcode(opcode).name


def run_matmul_demo(M, N, K, array_dim, any_layout=False, verbose=True):
    print("=" * 70)
    print(f"ISA Matmul: C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}] on a {array_dim}x{array_dim} array")
    print("=" * 70)

    config = SystolicConfig(
        grid_rows=array_dim, grid_cols=array_dim, sp_capacity_kb=4, acc_capacity_kb=4
    )
    beat = config.dma_buswidth // 8
    # Dense A and B, with C rows holding whole N tiles; or one row per DMA beat
    dense = (K, N, -(-N // array_dim) * array_dim * 4)
    strides = dense if any_layout else (beat, beat, beat)
    print(f"   strides (bytes): A={strides[0]} B={strides[1]} C={strides[2]}; DMA beat = {beat}")

    matmul = Matmul(config, require_beat_aligned=not any_layout)
    trace, status = [], {}

    async def controller(ctx):
        """Accept every command and complete it two cycles later."""
        countdown = None
        while True:
            ready = countdown is None
            ctx.set(matmul.cmd_ready, ready)
            ctx.set(matmul.cmd_completed, countdown == 0)
            _, _, valid, ctype, opcode, dram, local, rs1, rd, k_dim = await ctx.tick().sample(
                matmul.cmd_valid, matmul.cmd_type, matmul.cmd_opcode, matmul.cmd_dram_addr,
                matmul.cmd_local_addr, matmul.cmd_rs1, matmul.cmd_rd, matmul.cmd_k_dim,
            )  # fmt: skip
            if not ready:
                countdown = None if countdown == 0 else countdown - 1
            if valid and ready:
                trace.append((command_name(ctype, opcode), dram, local, rs1, rd, k_dim))
                countdown = 2

    async def testbench(ctx):
        config_cmds = [
            (MatmulCmd.CONFIG_DIMS, M | N << 16 | K << 32),
            (MatmulCmd.CONFIG_A_ADDR, A_ADDR),
            (MatmulCmd.CONFIG_B_ADDR, B_ADDR),
            (MatmulCmd.CONFIG_C_ADDR, C_ADDR),
            (MatmulCmd.CONFIG_A_STRIDE, strides[0]),
            (MatmulCmd.CONFIG_B_STRIDE, strides[1]),
            (MatmulCmd.CONFIG_C_STRIDE, strides[2]),
            (MatmulCmd.START, 0),
        ]
        for cmd, data in config_cmds:
            ctx.set(matmul.cfg_valid, 1)
            ctx.set(matmul.cfg_cmd, cmd)
            ctx.set(matmul.cfg_data, data)
            await ctx.tick()
        ctx.set(matmul.cfg_valid, 0)
        done, error, code = (
            await ctx.tick()
            .sample(matmul.done, matmul.error, matmul.error_code)
            .until(matmul.done | matmul.error)
        )
        status.update(done=done, error=error, code=code)

    sim = Simulator(matmul)
    sim.add_clock(1e-6)
    sim.add_testbench(testbench)
    sim.add_process(controller)
    sim.run()

    if status["error"]:
        print(f"\n   Matmul reported {MatmulError(status['code']).name}")
        if not any_layout:
            print("   (the tile's DMA cannot move this layout; try --any-layout)")
        return False

    if verbose:
        print("\n   Command stream:")
        for name, dram, local, rs1, rd, k_dim in trace:
            if name in ("LOAD", "STORE"):
                print(f"      {name:<18} dram=0x{dram:08X} local=0x{local:08X}")
            else:
                print(f"      {name:<18} rs1=0x{rs1:X} rd=0x{rd:X} k_dim={k_dim}")

    def tiles(n):
        return [min(array_dim, n - s) for s in range(0, n, array_dim)]

    expected = {
        "CONFIG": 1,
        "LOAD": 0,
        "STORE": 0,
        "PRELOAD": 0,
        "COMPUTE": 0,
        "COMPUTE_ACCUMULATE": 0,
    }
    for tm in tiles(M):
        for _tn in tiles(N):
            for n, tk in enumerate(tiles(K)):
                expected["LOAD"] += tm + tk
                expected["PRELOAD"] += 1
                expected["COMPUTE" if n == 0 else "COMPUTE_ACCUMULATE"] += 1
            expected["STORE"] += tm
    actual = {name: sum(1 for t in trace if t[0] == name) for name in expected}

    print("\n   Expected vs actual:")
    ok = True
    for name in expected:
        match = expected[name] == actual[name]
        ok &= match
        print(
            f"      {name:<18} {expected[name]:4d} {actual[name]:4d}  {'OK' if match else 'MISMATCH'}"
        )
    print("\n" + ("Demo completed successfully!" if ok else "Demo completed with mismatches."))
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--M", type=int, default=10, help="Rows of C (default: 10)")
    parser.add_argument("--N", type=int, default=4, help="Columns of C (default: 4)")
    parser.add_argument("--K", type=int, default=4, help="Inner dimension (default: 4)")
    parser.add_argument("--array-dim", type=int, default=4, help="Array dimension (default: 4)")
    parser.add_argument(
        "--any-layout", action="store_true", help="Dense operands, alignment check off"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Do not print the command stream"
    )
    args = parser.parse_args()
    ok = run_matmul_demo(args.M, args.N, args.K, args.array_dim, args.any_layout, not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
