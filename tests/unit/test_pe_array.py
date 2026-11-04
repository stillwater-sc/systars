"""Unit tests for PEArray: a block of PEs, one register per link, no timing role of its own."""

import numpy as np
import pytest

from systars.config import SystolicConfig
from systars.core.pe_array import EAST, NORTH, SOUTH, PEArray

from ._fabric import Schedule, column_clocks, column_values, play
from ._sim import skip_without_yosys

SHAPES = [(1, 1), (2, 2), (2, 3), (3, 2), (4, 4)]


def tile(rows, cols):
    return PEArray(SystolicConfig(tile_rows=rows, tile_cols=cols))


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_ports(rows, cols):
    dut = tile(rows, cols)
    for i in range(rows):
        for name in EAST:
            assert hasattr(dut, f"in_{name}_{i}") and hasattr(dut, f"out_{name}_{i}")
    for j in range(cols):
        for name in SOUTH + NORTH:
            assert hasattr(dut, f"in_{name}_{j}") and hasattr(dut, f"out_{name}_{j}")
        assert hasattr(dut, f"out_psum_valid_{j}")


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_output_stationary(rows, cols):
    """Staggered streams meet at every PE; results leave north, row 0 first."""
    dut = tile(rows, cols)
    rng = np.random.default_rng(rows * 10 + cols)
    k_len = 4
    A = rng.integers(-128, 128, (rows, k_len))
    B = rng.integers(-128, 128, (k_len, cols))
    sched = Schedule()
    sched.os_stream(0, A, B)
    trace = play(dut, cols, sched, stationary=False, extra=3 * rows + cols)

    np.testing.assert_array_equal(column_values(trace.north, cols), A @ B)
    expected = [[k_len + j + 2 * i for j in range(cols)] for i in range(rows)]
    np.testing.assert_array_equal(column_clocks(trace.north, cols), np.array(expected))


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_stationary(rows, cols):
    """Weights shift in last row first; A @ W leaves the south edge, one PE per clock."""
    dut = tile(rows, cols)
    rng = np.random.default_rng(rows * 10 + cols + 1)
    A = rng.integers(-128, 128, (5, rows))
    W = rng.integers(-128, 128, (rows, cols))
    sched = Schedule()
    sched.stationary_stream(sched.preload(0, W), A)
    trace = play(dut, cols, sched, stationary=True, extra=rows + cols + 2)

    np.testing.assert_array_equal(column_values(trace.south, cols), A @ W)


def test_verilog():
    skip_without_yosys()
    from amaranth.back import verilog

    assert "module PEArray" in verilog.convert(tile(2, 2), name="PEArray")
