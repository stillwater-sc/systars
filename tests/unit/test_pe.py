"""Unit tests for the PE: output-stationary MAC, weight-stationary MAC, shift chain."""

import pytest

from systars.config import SystolicConfig
from systars.core.pe import OUTPUT_STATIONARY, PE, WEIGHT_STATIONARY

from ._sim import run_sim, skip_without_yosys


@pytest.fixture
def pe():
    return PE(SystolicConfig())


def test_ports(pe):
    cfg = pe.config
    assert pe.in_a.shape().width == cfg.input_bits
    assert pe.in_b.shape().width == cfg.weight_bits
    for name in ("in_psum", "in_d", "out_psum", "out_c"):
        assert getattr(pe, name).shape().width == cfg.acc_bits
    for name in ("in_valid", "in_shift", "in_control_dataflow"):
        assert getattr(pe, name).shape().width == 1


def test_operands_pass_through_combinationally(pe):
    async def tb(ctx):
        ctx.set(pe.in_a, -7)
        ctx.set(pe.in_b, 13)
        assert ctx.get(pe.out_a) == -7
        assert ctx.get(pe.out_b) == 13

    run_sim(pe, tb)


def test_output_stationary_accumulates_when_valid(pe):
    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, OUTPUT_STATIONARY)
        total = 0
        for a, b, valid in [(3, 4, 1), (-5, 6, 1), (100, 100, 0), (-128, -128, 1)]:
            ctx.set(pe.in_a, a)
            ctx.set(pe.in_b, b)
            ctx.set(pe.in_valid, valid)
            await ctx.tick()
            total += a * b if valid else 0
            assert ctx.get(pe.out_c) == total
        assert total == 3 * 4 - 5 * 6 + 128 * 128

    run_sim(pe, tb)


def test_output_stationary_shift_loads_chain(pe):
    """Shifting replaces c with in_d (draining the result down the column)."""

    async def tb(ctx):
        ctx.set(pe.in_a, 2)
        ctx.set(pe.in_b, 3)
        ctx.set(pe.in_valid, 1)
        await ctx.tick()
        assert ctx.get(pe.out_c) == 6
        ctx.set(pe.in_valid, 0)
        ctx.set(pe.in_d, -99)
        ctx.set(pe.in_shift, 1)
        await ctx.tick()
        assert ctx.get(pe.out_c) == -99

    run_sim(pe, tb)


def test_shift_takes_priority_over_valid(pe):
    async def tb(ctx):
        ctx.set(pe.in_a, 5)
        ctx.set(pe.in_b, 5)
        ctx.set(pe.in_valid, 1)
        ctx.set(pe.in_shift, 1)
        ctx.set(pe.in_d, 1)
        await ctx.tick()
        assert ctx.get(pe.out_c) == 1

    run_sim(pe, tb)


def test_weight_stationary(pe):
    """WS: shifting loads the weight (truncated to weight_bits); psum_out = psum_in + a * w."""

    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, WEIGHT_STATIONARY)
        ctx.set(pe.in_d, -3 + (1 << 12))  # upper bits are dropped
        ctx.set(pe.in_shift, 1)
        await ctx.tick()
        ctx.set(pe.in_shift, 0)
        assert ctx.get(pe.out_c) == -3
        for a, psum in [(7, 0), (-128, 1000), (127, -5)]:
            ctx.set(pe.in_a, a)
            ctx.set(pe.in_psum, psum)
            ctx.set(pe.in_b, 99)  # ignored in WS
            assert ctx.get(pe.out_psum) == psum + a * -3

    run_sim(pe, tb)


def test_weight_stationary_valid_leaves_accumulator(pe):
    async def tb(ctx):
        ctx.set(pe.in_control_dataflow, WEIGHT_STATIONARY)
        ctx.set(pe.in_a, 4)
        ctx.set(pe.in_valid, 1)
        await ctx.tick()
        ctx.set(pe.in_control_dataflow, OUTPUT_STATIONARY)
        ctx.set(pe.in_valid, 0)
        assert ctx.get(pe.out_c) == 0

    run_sim(pe, tb)


def test_verilog(pe):
    skip_without_yosys()
    from amaranth.back import verilog

    assert "module PE" in verilog.convert(pe, name="PE")
