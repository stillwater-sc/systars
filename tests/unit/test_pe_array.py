"""Unit tests for PEArray: a combinational tile of PEs computing a block of A @ B."""

import numpy as np
import pytest

from systars.config import SystolicConfig
from systars.core.pe import OUTPUT_STATIONARY, WEIGHT_STATIONARY
from systars.core.pe_array import PEArray

from ._sim import run_sim, skip_without_yosys

SHAPES = [(1, 1), (2, 2), (2, 3), (3, 2), (4, 4)]


def tile(rows, cols):
    return PEArray(SystolicConfig(tile_rows=rows, tile_cols=cols))


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_ports(rows, cols):
    dut = tile(rows, cols)
    for i in range(rows):
        assert hasattr(dut, f"in_a_{i}") and hasattr(dut, f"out_a_{i}")
    for j in range(cols):
        for name in ("in_b", "in_psum", "in_d", "out_b", "out_psum", "out_c"):
            assert hasattr(dut, f"{name}_{j}")


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_output_stationary(rows, cols):
    """A and B in the same cycle meet at every PE; draining yields A @ B, bottom row first."""
    dut = tile(rows, cols)
    rng = np.random.default_rng(rows * 10 + cols)
    k = 4
    A = rng.integers(-128, 128, (rows, k))
    B = rng.integers(-128, 128, (k, cols))

    async def tb(ctx):
        ctx.set(dut.in_control_dataflow, OUTPUT_STATIONARY)
        ctx.set(dut.in_valid, 1)
        for step in range(k):
            for i in range(rows):
                ctx.set(getattr(dut, f"in_a_{i}"), int(A[i, step]))
            for j in range(cols):
                ctx.set(getattr(dut, f"in_b_{j}"), int(B[step, j]))
            await ctx.tick()
        ctx.set(dut.in_valid, 0)
        ctx.set(dut.in_shift, 1)
        drained = []
        for _ in range(rows):
            drained.append([ctx.get(getattr(dut, f"out_c_{j}")) for j in range(cols)])
            await ctx.tick()
        np.testing.assert_array_equal(np.array(drained[::-1]), A @ B)
        assert all(ctx.get(getattr(dut, f"out_c_{j}")) == 0 for j in range(cols))

    run_sim(dut, tb)


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_weight_stationary(rows, cols):
    """Weights shift in last row first; the partial sums give a @ W combinationally."""
    dut = tile(rows, cols)
    rng = np.random.default_rng(rows * 10 + cols + 1)
    W = rng.integers(-128, 128, (rows, cols))

    async def tb(ctx):
        ctx.set(dut.in_control_dataflow, WEIGHT_STATIONARY)
        ctx.set(dut.in_shift, 1)
        for r in reversed(range(rows)):
            for j in range(cols):
                ctx.set(getattr(dut, f"in_d_{j}"), int(W[r, j]))
            await ctx.tick()
        ctx.set(dut.in_shift, 0)
        for _ in range(3):
            a = rng.integers(-128, 128, rows)
            psum_in = rng.integers(-1000, 1000, cols)
            for i in range(rows):
                ctx.set(getattr(dut, f"in_a_{i}"), int(a[i]))
            for j in range(cols):
                ctx.set(getattr(dut, f"in_psum_{j}"), int(psum_in[j]))
            out = [ctx.get(getattr(dut, f"out_psum_{j}")) for j in range(cols)]
            np.testing.assert_array_equal(out, psum_in + a @ W)

    run_sim(dut, tb)


def test_verilog():
    skip_without_yosys()
    from amaranth.back import verilog

    assert "module PEArray" in verilog.convert(tile(2, 2), name="PEArray")
