"""Cocotb tests for the PE (default config): OS accumulate and drain, WS weight and psum.

Every ``cycle`` (drive inputs, then sample in the read-only phase) is followed
by a ``tick`` (the rising edge), so registered outputs are sampled one cycle
after the inputs that update them.
"""

import random

from tb.gemm import cycle, signed, start, tick

import cocotb

OS, WS = 0, 1


@cocotb.test()
async def test_pe_output_stationary(dut):
    """c accumulates a * b while valid; a shift replaces c with in_d."""
    await start(dut)
    rng = random.Random(1)
    total = 0
    for _ in range(8):
        a, b, valid = rng.randint(-128, 127), rng.randint(-128, 127), rng.random() < 0.75
        await cycle(dut, in_control_dataflow=OS, in_a=a, in_b=b, in_valid=valid, in_shift=0)
        assert signed(dut, "out_c") == total  # before this cycle's product
        assert signed(dut, "out_a") == a and signed(dut, "out_b") == b  # combinational
        await tick(dut)
        total += a * b if valid else 0
    await cycle(dut, in_valid=0, in_shift=1, in_d=-12345)
    assert signed(dut, "out_c") == total
    await tick(dut)
    await cycle(dut, in_shift=0)
    assert signed(dut, "out_c") == -12345
    await tick(dut)


@cocotb.test()
async def test_pe_weight_stationary(dut):
    """A shift loads the weight; out_psum = in_psum + a * w, combinationally."""
    await start(dut)
    await cycle(dut, in_control_dataflow=WS, in_valid=0, in_shift=1, in_d=-77)
    await tick(dut)
    rng = random.Random(2)
    for _ in range(8):
        a, psum = rng.randint(-128, 127), rng.randint(-100_000, 100_000)
        await cycle(dut, in_shift=0, in_a=a, in_psum=psum, in_b=rng.randint(-128, 127))
        assert signed(dut, "out_c") == -77
        assert signed(dut, "out_psum") == psum + a * -77
        await tick(dut)
