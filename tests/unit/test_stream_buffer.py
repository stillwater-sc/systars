"""Unit tests for L1: StreamBuffers (one block per buffer, one credit) and ResultBuffers."""

import numpy as np
import pytest

from systars.memory.stream_buffer import ResultBuffers, StreamBuffers

from ._sim import run_sim, skip_without_yosys

EB = 8


def pack(values, bits):
    word = 0
    for n, v in enumerate(values):
        word |= (int(v) & ((1 << bits) - 1)) << (n * bits)
    return word


def unpack(word, lanes, bits):
    out = []
    for n in range(lanes):
        v = (word >> (n * bits)) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >> (bits - 1) else v)
    return out


def load(ctx, dut, index, values, *, length=None, first=0, last=0):
    ctx.set(dut.load_valid, 1)
    ctx.set(dut.load_index, index)
    ctx.set(dut.load_data, pack(values, EB))
    ctx.set(dut.load_len, len(values) if length is None else length)
    ctx.set(dut.load_first, first)
    ctx.set(dut.load_last, last)


def idle(ctx, dut):
    ctx.set(dut.load_valid, 0)


def sample(ctx, dut, lane):
    if not ctx.get(getattr(dut, f"out_valid_{lane}")):
        return None
    return (
        ctx.get(getattr(dut, f"out_{lane}")),
        ctx.get(getattr(dut, f"out_first_{lane}")),
        ctx.get(getattr(dut, f"out_last_{lane}")),
    )


@pytest.mark.parametrize("lanes", [1, 4])
def test_streams_one_element_per_clock_after_the_load(lanes):
    dut = StreamBuffers(lanes, lanes, EB)
    block = [-3, 7, -128, 127][:lanes]

    async def tb(ctx):
        assert all(ctx.get(getattr(dut, f"credit_{i}")) for i in range(lanes))
        load(ctx, dut, lanes - 1, block, first=1, last=1)
        assert sample(ctx, dut, lanes - 1) is None  # nothing in the clock of the load
        await ctx.tick()
        idle(ctx, dut)
        got = []
        for _ in range(lanes + 2):
            got.append(sample(ctx, dut, lanes - 1))
            assert ctx.get(dut.busy) == (got[-1] is not None)
            await ctx.tick()
        expected = [(v, int(n == 0), int(n == lanes - 1)) for n, v in enumerate(block)]
        assert got == expected + [None, None]
        assert ctx.get(dut.error) == 0

    run_sim(dut, tb)


def test_back_to_back_blocks_have_no_gap():
    """A load in the clock the last element leaves streams the next block straight on."""
    lanes = 4
    dut = StreamBuffers(lanes, lanes, EB)
    blocks = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    async def tb(ctx):
        got = []
        load(ctx, dut, 0, blocks[0], first=1)
        await ctx.tick()
        n = 1
        for _ in range(3 * lanes + 1):
            # load exactly when the last element of the current block leaves
            leaving_last = ctx.get(dut.out_valid_0) and ctx.get(dut.credit_0)
            if n < len(blocks) and leaving_last:
                load(ctx, dut, 0, blocks[n], last=n == len(blocks) - 1)
                n += 1
            else:
                idle(ctx, dut)
            s = sample(ctx, dut, 0)
            if s is not None:
                got.append(s)
            await ctx.tick()
        values = [v for v, _, _ in got]
        assert values == [v for b in blocks for v in b]
        assert [f for _, f, _ in got] == [1] + [0] * 11
        assert [lst for _, _, lst in got] == [0] * 11 + [1]
        assert ctx.get(dut.error) == 0

    run_sim(dut, tb)


def test_load_without_credit_is_an_error():
    dut = StreamBuffers(4, 4, EB)

    async def tb(ctx):
        load(ctx, dut, 2, [1, 2, 3, 4])
        await ctx.tick()
        assert ctx.get(dut.credit_2) == 0 and ctx.get(dut.credit_1) == 1
        load(ctx, dut, 1, [1, 2, 3, 4])  # another buffer: fine
        await ctx.tick()
        assert ctx.get(dut.error) == 0
        load(ctx, dut, 2, [5, 6, 7, 8])  # two elements still to go
        await ctx.tick()
        idle(ctx, dut)
        await ctx.tick()
        assert ctx.get(dut.error) == 1

    run_sim(dut, tb)


def test_short_and_empty_blocks():
    """load_len shortens a block (an edge tile's K); zero streams nothing."""
    dut = StreamBuffers(4, 4, EB)

    async def tb(ctx):
        load(ctx, dut, 0, [9, 8, 7, 6], length=2, first=1, last=1)
        await ctx.tick()
        idle(ctx, dut)
        got = [sample(ctx, dut, 0)]
        await ctx.tick()
        got.append(sample(ctx, dut, 0))
        await ctx.tick()
        got.append(sample(ctx, dut, 0))
        assert got == [(9, 1, 0), (8, 0, 1), None]
        load(ctx, dut, 0, [1, 1, 1, 1], length=0)
        await ctx.tick()
        idle(ctx, dut)
        assert sample(ctx, dut, 0) is None and ctx.get(dut.busy) == 0

    run_sim(dut, tb)


def play_results(dut, lanes, arrivals, clocks):
    """``arrivals[t] = {column: value}``; returns the rows emitted, with their clocks."""
    rows = []

    async def tb(ctx):
        for t in range(clocks):
            now = arrivals.get(t, {})
            for j in range(lanes):
                ctx.set(getattr(dut, f"in_valid_{j}"), j in now)
                ctx.set(getattr(dut, f"in_{j}"), now.get(j, 0))
            if ctx.get(dut.out_valid):
                rows.append((t, unpack(ctx.get(dut.out_data), lanes, 32)))
            await ctx.tick()
        assert ctx.get(dut.error) == 0
        assert ctx.get(dut.busy) == 0

    run_sim(dut, tb)
    return rows


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("pattern", ["north", "south"])
def test_result_buffers_turn_columns_into_rows(lanes, pattern):
    """North: column j's row i at clock j + 2i (OS). South: at clock j + i (stationary)."""
    dut = ResultBuffers(lanes, lanes, 32)
    rng = np.random.default_rng(lanes)
    C = rng.integers(-(2**31), 2**31, (lanes, lanes))
    step = 2 if pattern == "north" else 1
    arrivals = {}
    for i in range(lanes):
        for j in range(lanes):
            arrivals.setdefault(j + step * i, {})[j] = int(C[i, j])
    rows = play_results(dut, lanes, arrivals, (step + 1) * lanes + 3)

    np.testing.assert_array_equal(np.array([r for _, r in rows]), C)
    # row i leaves the clock after the last column delivers it
    assert [t for t, _ in rows] == [lanes - 1 + step * i + 1 for i in range(lanes)]


def test_result_buffers_back_to_back_tiles():
    """The next tile's results arrive 2R - 1 clocks after the previous tile's (OS)."""
    lanes = 4
    dut = ResultBuffers(lanes, lanes, 32)
    rng = np.random.default_rng(7)
    tiles = [rng.integers(-1000, 1000, (lanes, lanes)) for _ in range(3)]
    arrivals = {}
    for n, C in enumerate(tiles):
        for i in range(lanes):
            for j in range(lanes):
                arrivals.setdefault(n * (2 * lanes - 1) + j + 2 * i, {})[j] = int(C[i, j])
    rows = play_results(dut, lanes, arrivals, 3 * (2 * lanes - 1) + 3 * lanes)
    np.testing.assert_array_equal(np.array([r for _, r in rows]), np.vstack(tiles))


def test_result_buffer_overflow_is_an_error():
    lanes = 2
    dut = ResultBuffers(lanes, lanes, 32)

    async def tb(ctx):
        ctx.set(dut.in_valid_0, 1)  # column 1 never delivers, so nothing drains
        for _ in range(lanes):
            assert ctx.get(dut.credit_0) == 1
            await ctx.tick()
        assert ctx.get(dut.credit_0) == 0
        await ctx.tick()
        ctx.set(dut.in_valid_0, 0)
        await ctx.tick()
        assert ctx.get(dut.error) == 1

    run_sim(dut, tb)


def test_verilog():
    skip_without_yosys()
    from amaranth.back import verilog

    assert "module StreamBuffers" in verilog.convert(StreamBuffers(4, 4, 8), name="StreamBuffers")
    assert "module ResultBuffers" in verilog.convert(ResultBuffers(4, 4, 32), name="ResultBuffers")
