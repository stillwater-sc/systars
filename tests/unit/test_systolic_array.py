"""
Unit tests for SystolicArray: a pipelined grid of PEArrays that skews its inputs
and deskews its outputs, so unskewed A and B vectors produce A @ B.
"""

import numpy as np
import pytest

from systars.config import SystolicConfig
from systars.core.pe import OUTPUT_STATIONARY, WEIGHT_STATIONARY
from systars.core.systolic_array import SystolicArray

from ._sim import run_sim, skip_without_yosys

# (grid_rows, grid_cols, tile_rows, tile_cols), including non-square grids and tiles
SHAPES = [
    (1, 1, 1, 1),
    (2, 2, 1, 1),
    (1, 1, 2, 2),
    (2, 2, 2, 2),
    (4, 4, 1, 1),
    (2, 3, 2, 1),
    (3, 2, 1, 2),
]


def array(grid_rows, grid_cols, tile_rows, tile_cols):
    cfg = SystolicConfig(
        grid_rows=grid_rows, grid_cols=grid_cols, tile_rows=tile_rows, tile_cols=tile_cols
    )
    return SystolicArray(cfg), grid_rows * tile_rows, grid_cols * tile_cols


@pytest.mark.parametrize("shape", SHAPES)
def test_latency(shape):
    dut, _, _ = array(*shape)
    assert dut.latency == (shape[0] - 1) + (shape[1] - 1)


@pytest.mark.parametrize("shape", SHAPES)
def test_control_reaches_output_after_latency(shape):
    dut, _, _ = array(*shape)

    async def tb(ctx):
        ctx.set(dut.in_valid, 1)
        ctx.set(dut.in_last, 1)
        for cycle in range(dut.latency + 2):
            expected = cycle == dut.latency
            assert ctx.get(dut.out_valid) == expected
            assert ctx.get(dut.out_last) == expected
            await ctx.tick()
            ctx.set(dut.in_valid, 0)
            ctx.set(dut.in_last, 0)

    run_sim(dut, tb)


@pytest.mark.parametrize("shape", SHAPES)
def test_output_stationary(shape):
    """Stream K steps of A columns and B rows, then drain C = A @ B, bottom row first."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape))
    k = 5
    A = rng.integers(-128, 128, (rows, k))
    B = rng.integers(-128, 128, (k, cols))

    async def tb(ctx):
        ctx.set(dut.in_control_dataflow, OUTPUT_STATIONARY)
        seen_last = False
        for step in range(k):
            for i in range(rows):
                ctx.set(getattr(dut, f"in_a_{i}"), int(A[i, step]))
            for j in range(cols):
                ctx.set(getattr(dut, f"in_b_{j}"), int(B[step, j]))
            ctx.set(dut.in_valid, 1)
            ctx.set(dut.in_last, step == k - 1)
            seen_last = ctx.get(dut.out_last)
            await ctx.tick()
        ctx.set(dut.in_valid, 0)
        ctx.set(dut.in_last, 0)
        while not seen_last:
            seen_last = ctx.get(dut.out_last)
            await ctx.tick()
        ctx.set(dut.in_shift, 1)
        drained = []
        for _ in range(rows):
            drained.append([ctx.get(getattr(dut, f"out_c_{j}")) for j in range(cols)])
            await ctx.tick()
        np.testing.assert_array_equal(np.array(drained[::-1]), A @ B)

    run_sim(dut, tb)


@pytest.mark.parametrize("shape", SHAPES)
def test_weight_stationary(shape):
    """Preload W (last row first), stream A rows; each row of A @ W leaves together."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 100)
    m_rows = 6
    A = rng.integers(-128, 128, (m_rows, rows))
    W = rng.integers(-128, 128, (rows, cols))

    async def tb(ctx):
        ctx.set(dut.in_control_dataflow, WEIGHT_STATIONARY)
        ctx.set(dut.in_shift, 1)
        for r in reversed(range(rows)):
            for j in range(cols):
                ctx.set(getattr(dut, f"in_d_{j}"), int(W[r, j]))
            await ctx.tick()
        ctx.set(dut.in_shift, 0)
        out = []
        for cycle in range(m_rows + dut.latency + 2):
            feeding = cycle < m_rows
            for i in range(rows):
                ctx.set(getattr(dut, f"in_a_{i}"), int(A[cycle, i]) if feeding else 0)
            ctx.set(dut.in_valid, feeding)
            ctx.set(dut.in_last, cycle == m_rows - 1)
            if ctx.get(dut.out_valid):
                out.append([ctx.get(getattr(dut, f"out_psum_{j}")) for j in range(cols)])
                assert ctx.get(dut.out_last) == (len(out) == m_rows)
            await ctx.tick()
        np.testing.assert_array_equal(np.array(out), A @ W)

    run_sim(dut, tb)


def test_verilog():
    skip_without_yosys()
    from amaranth.back import verilog

    dut, _, _ = array(2, 2, 2, 2)
    assert "module SystolicArray" in verilog.convert(dut, name="SystolicArray")
