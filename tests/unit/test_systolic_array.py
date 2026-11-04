"""
Unit tests for SystolicArray: a grid of PEArrays with a register on every PE link.

Streams are staggered by the caller (the L1 stream buffers), so these tests play
the schedule the Streamer will: row ``i`` starts one clock after row ``i - 1``.
"""

import numpy as np
import pytest

from systars.config import SystolicConfig
from systars.core.systolic_array import SystolicArray

from ._fabric import Schedule, column_clocks, column_values, play
from ._sim import skip_without_yosys

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
    dut = SystolicArray(cfg)
    return dut, dut.rows, dut.cols


def operands(rng, rows, k_len, cols):
    return rng.integers(-128, 128, (rows, k_len)), rng.integers(-128, 128, (k_len, cols))


@pytest.mark.parametrize("shape", SHAPES)
def test_dimensions(shape):
    dut, rows, cols = array(*shape)
    assert (rows, cols) == (shape[0] * shape[2], shape[1] * shape[3])


@pytest.mark.parametrize("shape", SHAPES)
def test_output_stationary_drains_north(shape):
    """One reduction: each column gets its results row 0 first, every other clock."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape))
    k_len, s = 5, 2
    A, B = operands(rng, rows, k_len, cols)
    sched = Schedule()
    sched.os_stream(s, A, B)
    trace = play(dut, cols, sched, stationary=False, extra=3 * rows + cols)

    np.testing.assert_array_equal(column_values(trace.north, cols), A @ B)
    expected = [[dut.os_result_clock(s, k_len, i, j) for j in range(cols)] for i in range(rows)]
    np.testing.assert_array_equal(column_clocks(trace.north, cols), np.array(expected))
    assert trace.collisions == []


@pytest.mark.parametrize("shape", SHAPES)
def test_output_stationary_back_to_back_after_the_drain(shape):
    """Short reductions (K < 2R - 1): the next tile starts 2R - 1 clocks later."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 1)
    k_len, tiles = 2, 3
    gap = max(k_len, 2 * rows - 1)
    sched, expected = Schedule(), []
    for n in range(tiles):
        A, B = operands(rng, rows, k_len, cols)
        sched.os_stream(1 + n * gap, A, B)
        expected.append(A @ B)
    trace = play(dut, cols, sched, stationary=False, extra=3 * rows + cols)

    assert trace.collisions == []
    np.testing.assert_array_equal(column_values(trace.north, cols), np.vstack(expected))


@pytest.mark.parametrize("shape", [s for s in SHAPES if s[0] * s[2] >= 2])
def test_output_stationary_gap_below_the_drain_collides(shape):
    """A gap of 2R - 2 clocks puts row R - 1's result and row 0's next launch on one link."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 2)
    sched = Schedule()
    for n in range(2):
        A, B = operands(rng, rows, 1, cols)
        sched.os_stream(1 + n * (2 * rows - 2), A, B)
    trace = play(dut, cols, sched, stationary=False, extra=3 * rows + cols)
    assert trace.collisions


@pytest.mark.parametrize("shape", SHAPES)
def test_output_stationary_long_reductions_need_no_gap(shape):
    """With K >= 2R - 1, tiles stream back to back: the drain is hidden."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 3)
    k_len, tiles = 2 * rows - 1, 3
    sched, expected = Schedule(), []
    for n in range(tiles):
        A, B = operands(rng, rows, k_len, cols)
        sched.os_stream(n * k_len, A, B)
        expected.append(A @ B)
    trace = play(dut, cols, sched, stationary=False, extra=3 * rows + cols)

    assert trace.collisions == []
    np.testing.assert_array_equal(column_values(trace.north, cols), np.vstack(expected))


@pytest.mark.parametrize("shape", SHAPES)
def test_output_stationary_reduction_spans_streams(shape):
    """first on the first stream and last on the second: C stays in the PEs between."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 4)
    A1, B1 = operands(rng, rows, 3, cols)
    A2, B2 = operands(rng, rows, 2, cols)
    sched = Schedule()
    sched.os_stream(0, A1, B1, last=False)
    sched.os_stream(3 + 4, A2, B2, first=False)  # a gap between the streams
    trace = play(dut, cols, sched, stationary=False, extra=3 * rows + cols)

    np.testing.assert_array_equal(column_values(trace.north, cols), A1 @ B1 + A2 @ B2)


@pytest.mark.parametrize("shape", SHAPES)
def test_stationary(shape):
    """Preload W, stream A: element m's partial sums leave the south edge, column j late."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 100)
    m_len = 6
    A = rng.integers(-128, 128, (m_len, rows))
    W = rng.integers(-128, 128, (rows, cols))
    sched = Schedule()
    s = sched.preload(0, W)
    sched.stationary_stream(s, A)
    trace = play(dut, cols, sched, stationary=True, extra=rows + cols + 2)

    np.testing.assert_array_equal(column_values(trace.south, cols), A @ W)
    expected = [[dut.stationary_result_clock(s, m, j) for j in range(cols)] for m in range(m_len)]
    np.testing.assert_array_equal(column_clocks(trace.south, cols), np.array(expected))
    assert trace.north == {}


@pytest.mark.parametrize("shape", SHAPES)
def test_stationary_streams_back_to_back(shape):
    """Two A tiles against the same weights, the second right after the first."""
    dut, rows, cols = array(*shape)
    rng = np.random.default_rng(sum(shape) + 101)
    A1 = rng.integers(-128, 128, (rows, rows))
    A2 = rng.integers(-128, 128, (3, rows))
    W = rng.integers(-128, 128, (rows, cols))
    sched = Schedule()
    s = sched.preload(0, W)
    sched.stationary_stream(s, A1)
    sched.stationary_stream(s + rows, A2)
    trace = play(dut, cols, sched, stationary=True, extra=rows + cols + 2)

    np.testing.assert_array_equal(column_values(trace.south, cols), np.vstack([A1, A2]) @ W)


def test_verilog():
    skip_without_yosys()
    from amaranth.back import verilog

    dut, _, _ = array(2, 2, 2, 2)
    assert "module SystolicArray" in verilog.convert(dut, name="SystolicArray")
