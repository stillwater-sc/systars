"""Cocotb tests for the PE (default config): registered links, the OS reduction and
its launch north, and the stationary mode.

Every ``cycle`` (drive inputs, then sample in the read-only phase) is followed
by a ``tick`` (the rising edge). Every input is registered, so a PE computes on
what it was given in the previous cycle.
"""

import random

from tb.gemm import cycle, signed, start, tick

import cocotb

OS, STATIONARY = 0, 1


@cocotb.test()
async def test_pe_links_are_registered(dut):
    await start(dut)
    await cycle(dut, in_control_dataflow=OS, in_a=-7, in_b=13, in_valid=0, in_c=99, in_c_valid=1)
    assert signed(dut, "out_a") == 0 and int(dut.out_c_valid.value) == 0
    await tick(dut)
    await cycle(dut, in_c_valid=0)
    assert signed(dut, "out_a") == -7 and signed(dut, "out_b") == 13
    assert signed(dut, "out_c") == 99 and int(dut.out_c_valid.value) == 1  # passing north
    await tick(dut)


@cocotb.test()
async def test_pe_output_stationary_reduction(dut):
    """first overwrites, products accumulate, last launches c + a * b north."""
    await start(dut)
    rng = random.Random(1)
    for _ in range(3):
        pairs = [(rng.randint(-128, 127), rng.randint(-128, 127)) for _ in range(5)]
        for n, (a, b) in enumerate(pairs):
            await cycle(dut, in_control_dataflow=OS, in_a=a, in_b=b, in_valid=1,
                        in_first=n == 0, in_last=n == len(pairs) - 1)  # fmt: skip
            await tick(dut)
        await cycle(dut, in_valid=0, in_first=0, in_last=0)
        assert int(dut.out_c_valid.value) == 1
        assert signed(dut, "out_c") == sum(a * b for a, b in pairs)
        assert int(dut.out_collision.value) == 0
        await tick(dut)


@cocotb.test()
async def test_pe_stationary(dut):
    """The chain loads w; out_psum = psum + a * w while a is valid."""
    await start(dut)
    await cycle(dut, in_control_dataflow=STATIONARY, in_valid=0, in_shift=1, in_w=-77)
    await tick(dut)
    rng = random.Random(2)
    for _ in range(8):
        a, psum = rng.randint(-128, 127), rng.randint(-100_000, 100_000)
        await cycle(dut, in_shift=0, in_a=a, in_psum=psum, in_valid=1, in_b=rng.randint(-128, 127))
        await tick(dut)
        await cycle(dut, in_valid=0)
        assert signed(dut, "out_w") == -77
        assert signed(dut, "out_psum") == psum + a * -77
        assert int(dut.out_c_valid.value) == 0
        await tick(dut)
