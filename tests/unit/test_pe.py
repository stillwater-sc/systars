"""Unit tests for the PE: registered links, the OS reduction and launch, stationary mode."""

import pytest

from systars.config import SystolicConfig
from systars.core.pe import OUTPUT_STATIONARY, PE, STATIONARY

from ._sim import run_sim, skip_without_yosys


@pytest.fixture
def pe():
    return PE(SystolicConfig())


def test_ports(pe):
    cfg = pe.config
    assert pe.in_a.shape().width == cfg.input_bits
    assert pe.in_b.shape().width == cfg.weight_bits
    assert pe.in_w.shape().width == cfg.weight_bits
    for name in ("in_psum", "in_c", "out_psum", "out_c"):
        assert getattr(pe, name).shape().width == cfg.acc_bits
    for name in ("in_valid", "in_first", "in_last", "in_c_valid", "in_shift", "out_collision"):
        assert getattr(pe, name).shape().width == 1


def test_every_link_is_registered(pe):
    """Each input reaches the neighbour one clock later, never in the same clock."""

    async def tb(ctx):
        values = {"a": -7, "valid": 1, "first": 1, "last": 0, "b": 13, "c": 1234, "c_valid": 1}
        for name, value in values.items():
            ctx.set(getattr(pe, f"in_{name}"), value)
        for name in ("a", "valid", "first", "b"):
            assert ctx.get(getattr(pe, f"out_{name}")) == 0
        assert ctx.get(pe.out_c_valid) == 0
        await ctx.tick()
        for name in ("a", "valid", "first", "last", "b"):
            assert ctx.get(getattr(pe, f"out_{name}")) == values[name]
        assert (ctx.get(pe.out_c), ctx.get(pe.out_c_valid)) == (1234, 1)

    run_sim(pe, tb)


async def _feed(ctx, pe, a, b, *, valid=1, first=0, last=0):
    """Present one product; it is computed in the next clock (from the registers)."""
    for name, value in {"a": a, "b": b, "valid": valid, "first": first, "last": last}.items():
        ctx.set(getattr(pe, f"in_{name}"), value)
    await ctx.tick()
    for name in ("valid", "first", "last"):
        ctx.set(getattr(pe, f"in_{name}"), 0)


def test_output_stationary_reduction_launches_north(pe):
    """first overwrites, then products accumulate; last launches c + a * b north."""

    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, OUTPUT_STATIONARY)
        await _feed(ctx, pe, 3, 4, first=1)
        await _feed(ctx, pe, -5, 6)
        await _feed(ctx, pe, 100, 100, valid=0)
        assert ctx.get(pe.out_c_valid) == 0
        await _feed(ctx, pe, -128, -128, last=1)
        assert ctx.get(pe.out_c_valid) == 1
        assert ctx.get(pe.out_c) == 3 * 4 - 5 * 6 + 128 * 128
        assert ctx.get(pe.out_collision) == 0
        await ctx.tick()
        assert ctx.get(pe.out_c_valid) == 0

    run_sim(pe, tb)


def test_output_stationary_first_needs_no_clear(pe):
    """A new reduction ignores what the last one left in c."""

    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, OUTPUT_STATIONARY)
        await _feed(ctx, pe, 9, 9, first=1)
        await _feed(ctx, pe, 2, 2)  # c = 85, never launched
        await _feed(ctx, pe, 2, 3, first=1)
        await _feed(ctx, pe, 1, 1, last=1)
        assert (ctx.get(pe.out_c_valid), ctx.get(pe.out_c)) == (1, 7)
        await _feed(ctx, pe, 5, -5, first=1, last=1)  # a one-product reduction
        assert (ctx.get(pe.out_c_valid), ctx.get(pe.out_c)) == (1, -25)

    run_sim(pe, tb)


def test_output_stationary_passes_results_north(pe):
    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, OUTPUT_STATIONARY)
        ctx.set(pe.in_c, -77)
        ctx.set(pe.in_c_valid, 1)
        await ctx.tick()
        ctx.set(pe.in_c_valid, 0)
        assert (ctx.get(pe.out_c), ctx.get(pe.out_c_valid)) == (-77, 1)
        await ctx.tick()
        assert ctx.get(pe.out_c_valid) == 0

    run_sim(pe, tb)


def test_output_stationary_collision(pe):
    """Launching while a result from the south is passing is flagged."""

    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, OUTPUT_STATIONARY)
        ctx.set(pe.in_c, 1)
        ctx.set(pe.in_c_valid, 1)
        await _feed(ctx, pe, 1, 1, first=1, last=1)
        assert ctx.get(pe.out_collision) == 1

    run_sim(pe, tb)


def test_stationary(pe):
    """The chain loads w; psum_out = psum_in + a * w while a is valid."""

    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, STATIONARY)
        ctx.set(pe.in_w, -3)
        ctx.set(pe.in_shift, 1)
        await ctx.tick()
        ctx.set(pe.in_shift, 0)
        assert ctx.get(pe.out_w) == -3
        for a, psum, valid in [(7, 0, 1), (-128, 1000, 1), (127, -5, 0)]:
            ctx.set(pe.in_a, a)
            ctx.set(pe.in_psum, psum)
            ctx.set(pe.in_valid, valid)
            ctx.set(pe.in_b, 99)  # ignored
            await ctx.tick()
            assert ctx.get(pe.out_psum) == psum + (a * -3 if valid else 0)
            assert ctx.get(pe.out_psum_valid) == valid
            assert ctx.get(pe.out_c_valid) == 0  # no OS launch in stationary mode

    run_sim(pe, tb)


def test_shift_holds_w_otherwise(pe):
    async def tb(ctx):
        ctx.set(pe.in_w, 5)
        ctx.set(pe.in_shift, 1)
        await ctx.tick()
        ctx.set(pe.in_shift, 0)
        ctx.set(pe.in_w, 6)
        await ctx.tick()
        assert ctx.get(pe.out_w) == 5

    run_sim(pe, tb)


def test_verilog(pe):
    skip_without_yosys()
    from amaranth.back import verilog

    assert "module PE" in verilog.convert(pe, name="PE")
