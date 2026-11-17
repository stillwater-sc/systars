"""
The result path (#98, ``controller/result_path.py``), the L1 -> L2 hop on its own: result
rows from the fabric's edge into the Combiner, by the descriptor the Streamer gave each
tile, and the result event once a tile's rows are in.
"""

from __future__ import annotations

from systars.config import Dataflow, SystolicConfig
from systars.controller.result_path import ResultPath
from systars.errors import L1Error

from ._sim import run_sim

C = 2


def make(dataflow=Dataflow.OUTPUT_STATIONARY):
    return ResultPath(SystolicConfig(grid_rows=C, grid_cols=C, dataflow=dataflow))


def describe(ctx, dut, *, rd, rows, ident, sync=0, hold=0, accumulate=0, epilogue=0):
    for name, value in (("rd", rd), ("rows", rows), ("id", ident), ("sync", sync),
                        ("hold", hold), ("accumulate", accumulate), ("epilogue", epilogue)):  # fmt: skip
        ctx.set(getattr(dut, f"tile_{name}"), value)
    ctx.set(dut.tile_valid, 1)


def row(ctx, dut, values, valid=1):
    for j, v in enumerate(values):
        ctx.set(getattr(dut, f"array_out_c_{j}"), v)
        ctx.set(getattr(dut, f"array_out_c_valid_{j}"), valid)


def test_a_tile_s_rows_go_to_its_combiner_rows_and_then_its_result():
    dut = make()
    writes, results, held = [], [], []

    async def monitor(ctx):
        async for _, _, req, addr, acc, res, rid, rsync, hd in ctx.tick().sample(
            dut.combiner_write_req, dut.combiner_write_addr, dut.combiner_accumulate,
            dut.result, dut.result_id, dut.result_sync, dut.held,
        ):  # fmt: skip
            if req:
                writes.append((addr, acc))
            if res:
                results.append((rid, rsync))
            held.append(hd)

    async def tb(ctx):
        assert ctx.get(dut.tile_ready) and not ctx.get(dut.busy)
        describe(ctx, dut, rd=40, rows=2, ident=7, sync=0x15, accumulate=1)
        await ctx.tick()
        ctx.set(dut.tile_valid, 0)
        assert ctx.get(dut.busy)  # a tile is queued
        for values in ((1, 2), (3, 4)):
            row(ctx, dut, values)
            await ctx.tick()
        row(ctx, dut, (0, 0), valid=0)
        await ctx.tick().repeat(4)
        assert not ctx.get(dut.busy) and not ctx.get(dut.error)

    run_sim(dut, tb, processes=[monitor])
    assert writes == [(40, 1), (41, 1)]
    assert results == [(7, 0x15)]  # once, after the tile's last row
    assert not any(held)  # a tracked command's tile completes nothing


def test_an_untracked_tile_completes_its_command_with_its_results():
    dut = make()
    held = []

    async def monitor(ctx):
        async for _, _, hd, hid in ctx.tick().sample(dut.held, dut.held_id):
            if hd:
                held.append(hid)

    async def tb(ctx):
        describe(ctx, dut, rd=0, rows=1, ident=3, hold=1)
        await ctx.tick()
        ctx.set(dut.tile_valid, 0)
        row(ctx, dut, (5, 6))
        await ctx.tick()
        row(ctx, dut, (0, 0), valid=0)
        await ctx.tick().repeat(4)

    run_sim(dut, tb, processes=[monitor])
    assert held == [3]


def test_a_row_without_a_tile_is_an_l1_error():
    dut = make()

    async def tb(ctx):
        row(ctx, dut, (1, 2))
        await ctx.tick()
        row(ctx, dut, (0, 0), valid=0)
        await ctx.tick().repeat(4)
        assert ctx.get(dut.error)
        assert ctx.get(dut.error_kind) == L1Error.RESULT_WITHOUT_COMMAND
        assert not ctx.get(dut.combiner_write_req)

    run_sim(dut, tb)


def test_a_north_collision_is_an_l1_error():
    dut = make()

    async def tb(ctx):
        ctx.set(dut.array_out_collision, 1)
        await ctx.tick()
        ctx.set(dut.array_out_collision, 0)
        await ctx.tick()
        assert ctx.get(dut.error_kind) == L1Error.NORTH_COLLISION

    run_sim(dut, tb)


def test_the_stationary_side_takes_the_partial_sums():
    dut = make(Dataflow.OUTPUT_STATIONARY | Dataflow.B_STATIONARY)
    writes = []

    async def monitor(ctx):
        async for _, _, req, addr in ctx.tick().sample(
            dut.combiner_write_req, dut.combiner_write_addr
        ):
            if req:
                writes.append(addr)

    async def tb(ctx):
        ctx.set(dut.stationary, 1)
        describe(ctx, dut, rd=8, rows=1, ident=1)
        await ctx.tick()
        ctx.set(dut.tile_valid, 0)
        row(ctx, dut, (9, 9))  # north results are ignored in the stationary mode
        await ctx.tick()
        row(ctx, dut, (0, 0), valid=0)
        await ctx.tick().repeat(3)
        assert writes == [] and ctx.get(dut.busy)  # still waiting for its row
        for j in range(C):
            ctx.set(getattr(dut, f"array_out_psum_{j}"), j)
            ctx.set(getattr(dut, f"array_out_psum_valid_{j}"), 1)
        await ctx.tick()
        for j in range(C):
            ctx.set(getattr(dut, f"array_out_psum_valid_{j}"), 0)
        await ctx.tick().repeat(3)

    run_sim(dut, tb, processes=[monitor])
    assert writes == [8]
